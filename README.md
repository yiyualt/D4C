1. You can download the data of AM_CDs, AM_Electronics, AM_Movies, TripAdvisor, Yelp: 
  
    https://drive.google.com/drive/folders/1NKdfAjshXVIyxy0HtnHU1-CP_coQq64J

Or you can refer to the following provided by Lei Li to process your own data at hand. 

    https://github.com/lileipisces/NETE 
    
and 

    https://github.com/lileipisces/Sentires-Guide 


2. Download Pretrained models: 

    https://drive.google.com/drive/folders/1UAdcvo2p5UHqamRLbGJOT0jRvNRN4_KG?usp=sharing


3. Run generate_counterfactual.py


    python generate_counterfactual.py --auxliary AM_Elctronics --target AM_CDs --learning_rate 1e-4 --epochs 50 --save_file "model.pth" --coef 0.5 --prob 0.3

4. Run train.py

   python train.py --auxliary AM_Elctronics --target AM_CDs --learning_rate 5e-4 --weight 0.1 --coef 0.1 --epochs 50


Code References: 
  - https://github.com/lileipisces/PETER
  - https://github.com/YuYi0/AdaRex
