 - 24 symbols to choose?
 
 -  hard coded 22lag --> change

- when train , arg for model name selection . 

refactor
- [x] add progress bar while training


- naming convention for models and the result of it to compare. like how many epochs are done.
  - [ ] how many epochs done
  - [ ] what kind of loss is used
  - [ ]
  
Questions
 - why the validation loss is smaller than the training loss?
 - QLIKE loss is not working > loss in nan

for evaluation
- i want my loss in only consider when the realized volatility is above a certain threshold. because when i use, i will use this prediction when the value is bigger than threshold.
- for the threshold, in rv 1h case, let's start with it's bigger than binance fee 0.0500%