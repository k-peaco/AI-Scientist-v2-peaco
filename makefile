execute-ai-scientist:
	python launch_scientist_bfts.py --load_ideas "/Users/kohei/Doc/AI-Scientist-v2-peaco/ai_scientist/ideas/i_cant_believe_its_not_better.json" --load_code  --add_dataset_ref --model_writeup o1-preview-2024-09-12 --model_citation gpt-4o-2024-11-20 --model_review gpt-4o-2024-11-20 --model_agg_plots o3-mini-2025-01-31 --num_cite_rounds 2 2>&1 | tee "output_$$(date +%Y%m%d_%H%M%S).log"
execute-ai-scientist-with-mynumber:
	python launch_scientist_bfts.py --load_ideas "ai_scientist/ideas/next_individual_number_card.json" --model_writeup o1-preview-2024-09-12 --model_citation gpt-4o-2024-11-20 --model_review gpt-4o-2024-11-20 --model_agg_plots o3-mini-2025-01-31 --num_cite_rounds 2 2>&1 | tee "output_$$(date +%Y%m%d_%H%M%S).log"
test-log_summarization.py:
	python -m ai_scientist.treesearch.log_summarization