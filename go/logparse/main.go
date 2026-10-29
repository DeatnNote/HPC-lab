package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync/atomic"
	"time"
)

type maskList []string

func (m *maskList) String() string     { return strings.Join(*m, ",") }
func (m *maskList) Set(v string) error { *m = append(*m, v); return nil }

// Default masks shared by all datasets: IPs and hex. Bare numeric tokens are masked in Preprocess.
var defaultMasks = []string{
	`(\d+\.){3}\d+(:\d+)?`,
	`0x[0-9a-fA-F]+`,
}

// Record is one parsed line written as JSONL.
type Record struct {
	Line      int               `json:"line"`
	Fields    map[string]string `json:"fields"`
	Content   string            `json:"content"`
	ClusterID int               `json:"cluster_id"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: logparse parse|serve [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "parse":
		runParse(os.Args[2:])
	case "serve":
		runServe(os.Args[2:])
	default:
		fmt.Fprintln(os.Stderr, "unknown command", os.Args[1])
		os.Exit(2)
	}
}

func commonFlags(fs *flag.FlagSet) (*int, *float64, *int, *maskList, *bool) {
	depth := fs.Int("depth", 4, "parse tree depth")
	st := fs.Float64("st", 0.5, "similarity threshold")
	maxChild := fs.Int("maxchild", 100, "max children per internal node")
	var masks maskList
	fs.Var(&masks, "mask", "extra regex to mask as <*> (repeatable)")
	noDefault := fs.Bool("no-default-masks", false, "disable the built-in masks")
	return depth, st, maxChild, &masks, noDefault
}

func buildDrain(depth int, st float64, maxChild int, masks maskList, noDefault bool) *Drain {
	all := append([]string{}, masks...)
	if !noDefault {
		all = append(all, defaultMasks...)
	}
	d, err := NewDrain(depth, st, maxChild, all)
	if err != nil {
		log.Fatal(err)
	}
	return d
}

func runParse(args []string) {
	fs := flag.NewFlagSet("parse", flag.ExitOnError)
	in := fs.String("in", "-", "input log file (- for stdin)")
	out := fs.String("out", "-", "output JSONL of parsed records")
	tmplOut := fs.String("templates", "", "output JSON of templates")
	format := fs.String("format", "<Content>", "LogHub-style log format")
	depth, st, maxChild, masks, noDefault := commonFlags(fs)
	fs.Parse(args)

	re, names, err := FormatRegex(*format)
	if err != nil {
		log.Fatal(err)
	}
	d := buildDrain(*depth, *st, *maxChild, *masks, *noDefault)

	var r io.Reader = os.Stdin
	if *in != "-" {
		f, err := os.Open(*in)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		r = f
	}
	var w io.Writer = os.Stdout
	if *out != "-" {
		f, err := os.Create(*out)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		w = f
	}
	bw := bufio.NewWriter(w)
	defer bw.Flush()
	enc := json.NewEncoder(bw)

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1024*1024), 16*1024*1024)
	n, skipped := 0, 0
	start := time.Now()
	for sc.Scan() {
		n++
		fields, ok := ExtractFields(re, names, sc.Text())
		if !ok {
			skipped++
			continue
		}
		content := fields["Content"]
		delete(fields, "Content")
		c := d.Add(content)
		if err := enc.Encode(Record{Line: n, Fields: fields, Content: content, ClusterID: c.ID}); err != nil {
			log.Fatal(err)
		}
	}
	if err := sc.Err(); err != nil {
		log.Fatal(err)
	}
	if *tmplOut != "" {
		writeTemplates(*tmplOut, d)
	}
	fmt.Fprintf(os.Stderr, "parsed %d lines (%d skipped) into %d templates in %s\n",
		n, skipped, len(d.Clusters()), time.Since(start).Round(time.Millisecond))
}

type templateJSON struct {
	ID       int    `json:"id"`
	Template string `json:"template"`
	Count    int    `json:"count"`
}

func templatesJSON(d *Drain) []templateJSON {
	cs := d.Clusters()
	out := make([]templateJSON, len(cs))
	for i, c := range cs {
		out[i] = templateJSON{ID: c.ID, Template: c.TemplateString(), Count: c.Count}
	}
	return out
}

func writeTemplates(path string, d *Drain) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", " ")
	if err := enc.Encode(templatesJSON(d)); err != nil {
		log.Fatal(err)
	}
}

// Server is the HTTP ingest service.
type Server struct {
	d        *Drain
	lines    atomic.Int64
	requests atomic.Int64
	errors   atomic.Int64
}

// NewMux wires the HTTP handlers (exported for tests).
func (s *Server) NewMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok\n") })
	mux.HandleFunc("/ingest", s.handleIngest)
	mux.HandleFunc("/templates", s.handleTemplates)
	mux.HandleFunc("/metrics", s.handleMetrics)
	return mux
}

type ingestResult struct {
	ClusterID int    `json:"cluster_id"`
	Template  string `json:"template"`
}

// handleIngest accepts newline-delimited message contents and returns one template per line.
func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	s.requests.Add(1)
	if r.Method != http.MethodPost {
		s.errors.Add(1)
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	sc := bufio.NewScanner(http.MaxBytesReader(w, r.Body, 8<<20))
	sc.Buffer(make([]byte, 1024*1024), 8<<20)
	var res []ingestResult
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		c := s.d.Add(line)
		s.lines.Add(1)
		res = append(res, ingestResult{ClusterID: c.ID, Template: strings.Join(c.Template, " ")})
	}
	if err := sc.Err(); err != nil {
		s.errors.Add(1)
		http.Error(w, err.Error(), http.StatusRequestEntityTooLarge)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(res)
}

func (s *Server) handleTemplates(w http.ResponseWriter, r *http.Request) {
	s.requests.Add(1)
	ts := templatesJSON(s.d)
	sort.Slice(ts, func(i, j int) bool { return ts[i].Count > ts[j].Count })
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ts)
}

// handleMetrics exposes counters in the Prometheus text format.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# TYPE logparse_lines_total counter\nlogparse_lines_total %d\n", s.lines.Load())
	fmt.Fprintf(w, "# TYPE logparse_requests_total counter\nlogparse_requests_total %d\n", s.requests.Load())
	fmt.Fprintf(w, "# TYPE logparse_errors_total counter\nlogparse_errors_total %d\n", s.errors.Load())
	fmt.Fprintf(w, "# TYPE logparse_templates gauge\nlogparse_templates %d\n", len(s.d.Clusters()))
}

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", ":8080", "listen address")
	depth, st, maxChild, masks, noDefault := commonFlags(fs)
	fs.Parse(args)
	s := &Server{d: buildDrain(*depth, *st, *maxChild, *masks, *noDefault)}
	srv := &http.Server{Addr: *addr, Handler: s.NewMux(), ReadHeaderTimeout: 5 * time.Second}
	log.Printf("logparse listening on %s", *addr)
	log.Fatal(srv.ListenAndServe())
}
