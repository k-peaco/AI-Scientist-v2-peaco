max-num-generations ?= 5
num-reflections ?= 3
execute-generate-research-ideas:
	python ai_scientist/perform_ideation_temp_free.py --workshop-file $(idea_md_file) --model gpt-4o-2024-05-13 --max-num-generations $(max-num-generations) --num-reflections $(num-reflections)

num_cite_rounds ?= 2
load_code ?= 0
add_dataset_ref ?= 0
execute-ai-scientist:
ifeq ($(load_code), 1)
	FLAGS += --load_code
endif
ifeq ($(add_dataset_ref), 1)
	FLAGS += --add_dataset_ref
endif
	python launch_scientist_bfts.py --load_ideas $(idea_json_file) --model_writeup gpt-4o-2024-11-20 --model_citation gpt-4o-2024-11-20 --model_review gpt-4o-2024-11-20 --model_agg_plots o3-mini-2025-01-31 --num_cite_rounds $(num_cite_rounds) 2>&1 $(FLAGS) | tee "output_$$(date +%Y%m%d_%H%M%S).log"



test-log_summarization.py:
	python -m ai_scientist.treesearch.log_summarization