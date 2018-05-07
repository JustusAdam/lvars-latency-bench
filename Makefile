#VERTICES   =  40000
#EDGES      = 320000

VERTICES    = 120000
EDGES       = 1000000

RAND_DATA  = /tmp/rand

PBBSDIR    = ../pbbs/testData/graphData

EXEC = stack exec --

GRAPH_DATA_FILE = $(RAND_DATA)_$(EDGES)_$(VERTICES)

default: rand_data run_benches

rand_data:
	(cd $(PBBSDIR); make randLocalGraph)
	$(PBBSDIR)/randLocalGraph -m $(EDGES) -d 5 $(VERTICES) $(GRAPH_DATA_FILE)

clean_data:
	-rm -f /tmp/rand*

run_benches:
	stack build
	$(EXEC) ohua-sbfm-latency $(GRAPH_DATA_FILE) 10 64 +RTS -N7
	$(EXEC) ohua-fbm-latency $(GRAPH_DATA_FILE) 10 64 +RTS -N7
	$(EXEC) LVar-latency $(GRAPH_DATA_FILE) 10 64 +RTS -N2

make_plot:
	python makeplot.py plot -o fig.png
