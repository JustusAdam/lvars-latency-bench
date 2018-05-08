#VERTICES   =  40000
#EDGES      = 320000

VERTICES    := 120000
EDGES       := 1000000

RAND_DATA  := /tmp/rand

PBBSDIR    := ../pbbs/testData/graphData

EXEC := stack exec --

DYNAMIC_GRAPH_DATA_FILE := $(RAND_DATA)_$(EDGES)_$(VERTICES)

STATIC_GRAPH_DATA_FILE := radon-graph

GRAPH_DATA_FILE := $(STATIC_GRAPH_DATA_FILE)

HIGH_CORES := 7
LOW_CORES := 7

default: run_benches

rand_data:
	(cd $(PBBSDIR); make randLocalGraph)
	$(PBBSDIR)/randLocalGraph -m $(EDGES) -d 5 $(VERTICES) $(GRAPH_DATA_FILE)

clean_data:
	-rm -f /tmp/rand*

run_benches:
	stack build
	$(EXEC) ohua-sbfm-latency $(GRAPH_DATA_FILE) 10 64 +RTS -N$(HIGH_CORES)
	$(EXEC) ohua-fbm-latency $(GRAPH_DATA_FILE) 10 64 +RTS -N$(HIGH_CORES)
	$(EXEC) monad-par-latency $(GRAPH_DATA_FILE) 10 64 +RTS -N$(HIGH_CORES)
	$(EXEC) strategies-latency $(GRAPH_DATA_FILE) 10 64 +RTS -N$(HIGH_CORES)
	$(EXEC) LVar-latency $(GRAPH_DATA_FILE) 10 64 +RTS -N$(HIGH_CORES)

make_plot:
	python makeplot.py plot -o fig.png
