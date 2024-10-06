from AdvTrain import * 
from datasets import Dataset
from config import get_task_config
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)

if __name__ == "__main__":
    seed = 3407
    torch.manual_seed(seed)
    for task_idx in range(1, 9):
        task_config = get_task_config(task_idx)
        auxiliary = task_config["auxiliary"]
        target = task_config["target"]
        device = "cuda:0"
        log_file = "save.log"
        save_file = str(task_idx)+"/model.pth"
        epochs = 50
        coef = task_config["coef"]
        learning_rate = task_config["lr"]
        adv = task_config["adv"]
        path = "../../Merged_data/" + str(task_idx)
        train_df = pd.read_csv(path + "/aug_train.csv")
        train_df['item'] = train_df['item'].astype(str)
        nuser = train_df['user_idx'].max() + 1
        nitem = train_df['item_idx'].max() + 1
        config = {
            "task_idx": task_idx,
            "device": device,
            "log_file": log_file,
            "save_file":save_file,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": 128,
            "emsize": 768,
            "nlayers": 2,
            "nhid": 2048,
            "ntoken": 32128,
            "dropout": 0.2,
            "nuser": nuser,
            "nitem": nitem, 
            "coef": coef, 
            "nhead": 2
            }
        # load domain, user, item profiles from auxliary 
        suser_profiles = torch.tensor(np.load("../../data/"+auxiliary+"/user_profiles.npy"), dtype=torch.float, device=config.get("device"))
        sitem_profiles = torch.tensor(np.load("../../data/"+auxiliary+"/item_profiles.npy"), dtype=torch.float, device=config.get("device"))
        sdomain_profiles = torch.tensor(np.load("../../data/"+auxiliary+"/domain.npy"), dtype=torch.float, device=config.get("device"))

        # load target
        tuser_profiles = torch.tensor(np.load("../../data/"+target+"/user_profiles.npy"), dtype=torch.float, device=config.get("device"))
        titem_profiles = torch.tensor(np.load("../../data/"+target+"/item_profiles.npy"), dtype=torch.float, device=config.get("device"))
        tdomain_profiles = torch.tensor(np.load("../../data/"+target+"/domain.npy"), dtype=torch.float, device=config.get("device"))

        domain_profiles = torch.cat([sdomain_profiles.unsqueeze(0), tdomain_profiles.unsqueeze(0)], dim=0)
        user_profiles = torch.cat([tuser_profiles, suser_profiles], dim=0)
        item_profiles = torch.cat([titem_profiles, sitem_profiles], dim=0)
        model = Model(config.get("nuser"), config.get("nitem"), config.get("ntoken"), config.get("emsize"), config.get("nhead"), config.get("nhid"), config.get("nlayers"), config.get("dropout"), user_profiles, item_profiles, domain_profiles).to(config.get("device"))
        model.load_state_dict(torch.load(config.get("save_file")))

        target_df = train_df[train_df['domain'] == 'target'].copy()
        target_df["domain"] = "auxiliary"
        target_dataset = Dataset.from_pandas(target_df)
        processor = Processor(auxiliary, target)
        encoded_data = target_dataset.map(lambda sample: processor(sample))
        encoded_data.set_format("torch")

        test_dataset = TensorDataset(
            encoded_data['user_idx'],
            encoded_data['item_idx'],
            encoded_data['rating'],
            encoded_data['explanation_idx'],
            encoded_data['domain_idx']
        )
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        model.eval()
        prediction_exps = []
        reference_exps = []
        entropy_values = []

        with torch.no_grad():
            for batch in test_dataloader:
                user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = model.gather(batch, device)
                pred_ratings = model.recommend(user_idx, item_idx, domain_idx)
                pred_exps, entropy = model.generate(user_idx, item_idx,  domain_idx)
                prediction_exps.extend(tokenizer.batch_decode(pred_exps, skip_special_tokens=True))
                reference_exps.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))
                entropy_values.extend(entropy.cpu().numpy())

        filtered_indices = filter_by_entropy(entropy_values)
        filtered_prediction_exps = [prediction_exps[i] for i in filtered_indices]
        filtered_target_df = target_df.iloc[filtered_indices].copy()
        filtered_target_df['explanation'] = filtered_prediction_exps

        updated_train_df = train_df[train_df['domain'] == 'target'].copy() 
        final_df = pd.concat([updated_train_df, filtered_target_df])
        final_df.to_csv(str(task_idx)+"/factuals_counterfactuals.csv", index=False)