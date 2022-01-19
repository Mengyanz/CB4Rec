
def inference_news(args, news_path):

    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        news_path, 
        args,
        tokenizer
    )

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
    x for x in
    [news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory]
    if x is not None], axis=1)


    class NewsDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return self.data.shape[0]

    def news_collate_fn(arr):
        arr = torch.LongTensor(arr)
        return arr

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                batch_size=args.eva_batch_size,
                                num_workers=args.num_workers,
                                collate_fn=news_collate_fn)

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            input_ids = input_ids.cuda()
            news_vec = self.simulator.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)


    news_scoring = np.array(news_scoring)

    logging.info("news scoring num: {}".format(news_scoring.shape[0]))

    return news_index, new_scoring


def load_plm_simulator(args):
    checkpoint = torch.load(args.sim_path)
    if 'subcategory_dict' in checkpoint:
        subcategory_dict = checkpoint['subcategory_dict']
    else:
        subcategory_dict = {}

    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']
    domain_dict = checkpoint['domain_dict']

    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from PLM4NewsRec.model_bert import ModelBert
    from PLM4NewsRec.preprocess import read_news, read_news_bert, get_doc_input, get_doc_input_bert
    from PLM4NewsRec.dataloader import DataLoaderTrain, DataLoaderTest
    from torch.utils.data import Dataset, DataLoader

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_model = AutoModel.from_pretrained("bert-base-uncased",config=config)
    self.simulator = ModelBert(args, bert_model, len(category_dict), len(domain_dict), len(subcategory_dict)).to(self.device)
    self.simulator.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {sim_path}")
    self.simulator.eval()

    news_index, new_scoring = inference_news(args, os.path.join(args.root_data_dir,
                f'{args.dataset}/train_valid/news.tsv'))


    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_scoring,
        word_dict=word_dict,
        news_bias_scoring= None,
        data_dir=os.path.join(args.root_data_dir,
                            f'{args.dataset}/{args.test_dir}'),
        filename_pat=args.filename_pat,
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=False,
        enable_gpu=args.enable_gpu,
    )

    scores = {}

    for cnt, (user_ids, log_vecs, log_mask, news_vecs, news_bias, labels) in enumerate(dataloader):
        his_lens = torch.sum(log_mask, dim=-1).to(torch.device("cpu")).detach().numpy()

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(non_blocking=True)
            log_mask = log_mask.cuda(non_blocking=True)

        user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()
        
        # for i, user_vec in enumerate(user_vecs):
        #     for news_vec in cand_news_vecs: # TODO: cand_news_vecs
        #         score = np.dot(news_vec, user_vec)
        #         scores[user_ids[i]][news_id] = score # TODO: news_id


def propensity_predictor(context, exists):
    """learn propensity score
    """
    pass


def ips_simulator():
    """reweigh training samplings using ips
    simulator for user-choice 
    """
    pass


def generate_rewards():
    """generate and save simulated rewards on testing data
    """
    pass
