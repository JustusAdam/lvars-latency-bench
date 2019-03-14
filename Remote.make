 ELM := 141.30.52.30
REMOTE_USER := jst
SERVER := $(ELM)
REMOTE := $(REMOTE_USER)@$(SERVER)
PROJECT_FOLDER := 'projects/lvars-latency-bench'
ZIP_FILE := 'results.zip'
COMMAND := "cd $(PROJECT_FOLDER) && make run_benches $(MAKEARGS)"
ZIPCOMMAND := "cd $(PROJECT_FOLDER) && python makeplot.py -o $(ZIP_FILE) zip $(ZIPARGS)"
PLOT_FILE := 'plot.png'
PLOTARGS :=
ZIPARGS :=
RUNARGS :=
DEPTH := 20
REPETITIONS := 25
RT_COMMAND := "cd $(PROJECT_FOLDER) && python makeplot.py run $(RUNARGS) -g radon-graph --depth $(DEPTH) -r $(REPETITIONS)"
RT_JSON := 'res-avg-rt.json'

default: run_rt_experiment get_rt_data gen_rt_plot


run_experiment:
	ssh $(REMOTE) $(COMMAND)

get_data:
	ssh $(REMOTE) $(ZIPCOMMAND)
	scp "$(REMOTE):$(PROJECT_FOLDER)/$(ZIP_FILE)" ./$(ZIP_FILE)

create_plot:
	python makeplot.py -o $(PLOT_FILE) plot --use-zipped-data $(ZIP_FILE) $(PLOTARGS)

run_rt_experiment:
	ssh $(REMOTE) $(RT_COMMAND)

get_rt_data:
	scp "$(REMOTE):$(PROJECT_FOLDER)/$(RT_JSON)" ./$(RT_JSON)

gen_rt_plot:
	python makeplot.py -o $(PLOT_FILE) plot-rt $(PLOTARGS)
