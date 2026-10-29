// Package main implements a Drain-style online log template miner
// (He et al., "Drain: An Online Log Parsing Approach with Fixed Depth Tree", ICWS 2017)
// plus a log-format extractor, a CLI and a small HTTP ingest service. Standard library only.
package main

import (
	"fmt"
	"regexp"
	"strings"
	"sync"
)

const wildcard = "<*>"

// Cluster is one log template (event type).
type Cluster struct {
	ID       int      `json:"id"`
	Template []string `json:"-"`
	Count    int      `json:"count"`
}

// TemplateString joins the template tokens.
func (c *Cluster) TemplateString() string { return strings.Join(c.Template, " ") }

type node struct {
	children map[string]*node
	clusters []*Cluster
}

func newNode() *node { return &node{children: map[string]*node{}} }

// Drain is a fixed-depth parse tree. Safe for concurrent use.
type Drain struct {
	mu       sync.Mutex
	depth    int     // tree depth including the root and length layers (>= 3)
	st       float64 // similarity threshold
	maxChild int
	root     *node
	clusters []*Cluster
	masks    []*regexp.Regexp
}

// NewDrain builds a parser. masks are regexes whose matches are replaced by <*> before parsing.
func NewDrain(depth int, st float64, maxChild int, masks []string) (*Drain, error) {
	if depth < 3 {
		return nil, fmt.Errorf("depth must be >= 3, got %d", depth)
	}
	d := &Drain{depth: depth, st: st, maxChild: maxChild, root: newNode()}
	for _, m := range masks {
		re, err := regexp.Compile(m)
		if err != nil {
			return nil, fmt.Errorf("bad mask %q: %w", m, err)
		}
		d.masks = append(d.masks, re)
	}
	return d, nil
}

// Preprocess applies the masks and tokenizes on whitespace.
func (d *Drain) Preprocess(content string) []string {
	for _, re := range d.masks {
		content = re.ReplaceAllString(content, wildcard)
	}
	toks := strings.Fields(content)
	for i, t := range toks {
		if numericToken.MatchString(t) {
			toks[i] = wildcard
		}
	}
	return toks
}

var numericToken = regexp.MustCompile(`^[-+]?\d+([.,:]\d+)*$`)

func hasDigit(s string) bool {
	for _, r := range s {
		if r >= '0' && r <= '9' {
			return true
		}
	}
	return false
}

// similarity returns the share of positions with identical tokens and the number of
// wildcard positions in the template (used as a tie-breaker, as in the original Drain).
func similarity(tmpl, tokens []string) (float64, int) {
	same, params := 0, 0
	for i, t := range tmpl {
		if t == wildcard {
			params++
			continue
		}
		if t == tokens[i] {
			same++
		}
	}
	return float64(same) / float64(len(tmpl)), params
}

// Add parses one message and returns the cluster it was assigned to.
func (d *Drain) Add(content string) *Cluster {
	tokens := d.Preprocess(content)
	if len(tokens) == 0 {
		tokens = []string{""}
	}
	d.mu.Lock()
	defer d.mu.Unlock()

	// Layer 1: message length.
	key := fmt.Sprintf("len=%d", len(tokens))
	cur, ok := d.root.children[key]
	if !ok {
		cur = newNode()
		d.root.children[key] = cur
	}
	// Layers 2..depth-1: leading tokens (tokens with digits go to the wildcard branch).
	for i := 0; i < d.depth-2 && i < len(tokens); i++ {
		tok := tokens[i]
		if hasDigit(tok) {
			tok = wildcard
		}
		next, ok := cur.children[tok]
		if !ok {
			// One slot per node is reserved for the wildcard branch once the node is full.
			if len(cur.children) < d.maxChild-1 {
				next = newNode()
				cur.children[tok] = next
			} else {
				if _, w := cur.children[wildcard]; !w {
					cur.children[wildcard] = newNode()
				}
				next = cur.children[wildcard]
			}
		}
		cur = next
	}

	// Leaf: most similar cluster.
	var best *Cluster
	bestSim, bestParams := -1.0, -1
	for _, c := range cur.clusters {
		sim, params := similarity(c.Template, tokens)
		if sim > bestSim || (sim == bestSim && params > bestParams) {
			best, bestSim, bestParams = c, sim, params
		}
	}
	if best != nil && bestSim >= d.st {
		for i, t := range best.Template {
			if t != tokens[i] {
				best.Template[i] = wildcard
			}
		}
		best.Count++
		return best
	}
	c := &Cluster{ID: len(d.clusters) + 1, Template: append([]string(nil), tokens...), Count: 1}
	d.clusters = append(d.clusters, c)
	cur.clusters = append(cur.clusters, c)
	return c
}

// Clusters returns a snapshot of all clusters.
func (d *Drain) Clusters() []Cluster {
	d.mu.Lock()
	defer d.mu.Unlock()
	out := make([]Cluster, len(d.clusters))
	for i, c := range d.clusters {
		out[i] = Cluster{ID: c.ID, Template: append([]string(nil), c.Template...), Count: c.Count}
	}
	return out
}

// FormatRegex turns a LogHub-style format such as "<Date> <Node> <Content>" into a regex
// with one named group per header.
func FormatRegex(format string) (*regexp.Regexp, []string, error) {
	hdr := regexp.MustCompile(`<[^<>]+>`)
	var sb strings.Builder
	var names []string
	sb.WriteString("^")
	last := 0
	for _, loc := range hdr.FindAllStringIndex(format, -1) {
		sep := format[last:loc[0]]
		sb.WriteString(regexp.MustCompile(` +`).ReplaceAllString(regexp.QuoteMeta(sep), `\s+`))
		name := format[loc[0]+1 : loc[1]-1]
		names = append(names, name)
		if name == "Content" {
			sb.WriteString(`(?P<Content>.*)`)
		} else {
			sb.WriteString(fmt.Sprintf(`(?P<%s>.*?)`, name))
		}
		last = loc[1]
	}
	sb.WriteString(regexp.QuoteMeta(format[last:]))
	sb.WriteString("$")
	re, err := regexp.Compile(sb.String())
	return re, names, err
}

// ExtractFields applies a format regex to one raw line.
func ExtractFields(re *regexp.Regexp, names []string, line string) (map[string]string, bool) {
	m := re.FindStringSubmatch(strings.TrimRight(line, "\r\n"))
	if m == nil {
		return nil, false
	}
	out := make(map[string]string, len(names))
	for i, n := range re.SubexpNames() {
		if n != "" {
			out[n] = strings.TrimSpace(m[i])
		}
	}
	return out, true
}
