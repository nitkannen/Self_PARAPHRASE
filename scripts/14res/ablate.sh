python main.py --task 14res  --train_dataset_path 14res/train  --dev_dataset_path 14res/dev  --test_dataset_path 14res/test  --model_name_or_path t5-base  --do_train  --do_eval  --train_batch_size 2  --gradient_accumulation_steps 2  --eval_batch_size 16  --learning_rate 1e-4  --num_train_epochs 20  --regressor True  --use_tagger True --logger_name ablate_14res_no_contrast.txt  --log_message epoch12_2_2_1e4_0.2default  --gpu_id 1 --seed 47

python main.py --task 14res  --train_dataset_path 14res/train  --dev_dataset_path 14res/dev  --test_dataset_path 14res/test  --model_name_or_path t5-base  --do_train  --do_eval  --train_batch_size 2  --gradient_accumulation_steps 2  --eval_batch_size 16  --learning_rate 1e-4  --num_train_epochs 20  --use_tagger True --logger_name ablate_14res_no_contrast_no_regressor.txt  --log_message epoch12_2_2_1e4_0.2default  --gpu_id 1 --seed 47

python main.py --task 14res  --train_dataset_path 14res/train  --dev_dataset_path 14res/dev  --test_dataset_path 14res/test  --model_name_or_path t5-base  --do_train  --do_eval  --train_batch_size 2  --gradient_accumulation_steps 2  --eval_batch_size 16  --learning_rate 1e-4  --num_train_epochs 20  --logger_name ablate_14res_no_contrast_no_regressor_no_tagger.txt  --log_message epoch12_2_2_1e4_0.2default  --gpu_id 1 --seed 47

python main.py --task lap14 --train_dataset_path lap14/train --dev_dataset_path lap14/dev --test_dataset_path lap14/test --model_name_or_path t5-base --do_train --do_eval --train_batch_size 2 --gradient_accumulation_steps 2 --eval_batch_size 16 --learning_rate 3e-4 --num_train_epochs 20 --regressor True --use_tagger True --beta 0.1  --logger_name ablate_lap14_no_contrast.txt  --log_message epoch10_2_2_3e4_0.1default --gpu_id 1 

python main.py --task lap14 --train_dataset_path lap14/train --dev_dataset_path lap14/dev --test_dataset_path lap14/test --model_name_or_path t5-base --do_train --do_eval --train_batch_size 2 --gradient_accumulation_steps 2 --eval_batch_size 16 --learning_rate 3e-4 --num_train_epochs 20 --use_tagger True --beta 0.1  --logger_name ablate_lap14_no_contrast_no_regressor.txt  --log_message epoch10_2_2_3e4_0.1default --gpu_id 1 

python main.py --task lap14 --train_dataset_path lap14/train --dev_dataset_path lap14/dev --test_dataset_path lap14/test --model_name_or_path t5-base --do_train --do_eval --train_batch_size 2 --gradient_accumulation_steps 2 --eval_batch_size 16 --learning_rate 3e-4 --num_train_epochs 20 --beta 0.1  --logger_name ablate_lap14_no_contrast_no_regressor_no_tagger.txt  --log_message epoch10_2_2_3e4_0.1default --gpu_id 1 