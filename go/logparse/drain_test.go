package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func mustDrain(t *testing.T) *Drain {
	t.Helper()
	d, err := NewDrain(4, 0.5, 100, defaultMasks)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestSameTemplateMerges(t *testing.T) {
	d := mustDrain(t)
	a := d.Add("Lustre: slow reply from OST0003 after 31s")
	b := d.Add("Lustre: slow reply from OST0007 after 45s")
	if a.ID != b.ID {
		t.Fatalf("expected one template, got %d and %d", a.ID, b.ID)
	}
	if got, want := b.TemplateString(), "Lustre: slow reply from <*> after <*>"; got != want {
		t.Fatalf("template = %q, want %q", got, want)
	}
}

func TestDifferentEventsStaySeparate(t *testing.T) {
	d := mustDrain(t)
	a := d.Add("kernel: EDAC MC0: 3 CE memory read error on DIMM 2")
	b := d.Add("kernel: link down on port 3 retraining now ok")
	if a.ID == b.ID {
		t.Fatal("unrelated messages of equal length were merged")
	}
}

func TestNumericAndHexMasked(t *testing.T) {
	d := mustDrain(t)
	toks := d.Preprocess("addr 0xdeadbeef from 10.1.2.3:22 took 17 ms")
	got := strings.Join(toks, " ")
	if want := "addr <*> from <*> took <*> ms"; got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestDepthValidation(t *testing.T) {
	if _, err := NewDrain(2, 0.5, 100, nil); err == nil {
		t.Fatal("depth 2 should be rejected")
	}
}

func TestFormatRegexBGL(t *testing.T) {
	re, names, err := FormatRegex("<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>")
	if err != nil {
		t.Fatal(err)
	}
	line := "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
	f, ok := ExtractFields(re, names, line)
	if !ok {
		t.Fatal("line did not match")
	}
	if f["Level"] != "INFO" || f["Content"] != "instruction cache parity error corrected" {
		t.Fatalf("bad fields: %+v", f)
	}
}

func TestConcurrentAddIsSafe(t *testing.T) {
	d := mustDrain(t)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				d.Add("job 12345 started on 64 nodes")
			}
		}(g)
	}
	wg.Wait()
	cs := d.Clusters()
	if len(cs) != 1 || cs[0].Count != 4000 {
		t.Fatalf("want 1 cluster with 4000 lines, got %+v", cs)
	}
}

func TestHTTPIngestAndMetrics(t *testing.T) {
	s := &Server{d: mustDrain(t)}
	srv := httptest.NewServer(s.NewMux())
	defer srv.Close()

	body := "fan FAN3 speed 400 RPM below lower critical threshold\nfan FAN5 speed 610 RPM below lower critical threshold\n"
	resp, err := http.Post(srv.URL+"/ingest", "text/plain", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if r, _ := http.Get(srv.URL + "/ingest"); r.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET /ingest should be 405, got %d", r.StatusCode)
	}
	m, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	buf := new(strings.Builder)
	_, _ = io.Copy(buf, m.Body)
	m.Body.Close()
	out := buf.String()
	for _, want := range []string{"logparse_lines_total 2", "logparse_templates 1", "logparse_errors_total 1"} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics missing %q:\n%s", want, out)
		}
	}
}
