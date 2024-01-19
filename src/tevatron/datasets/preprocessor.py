VAR_PREFIX = "[VAR]"


class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32,
                 text_max_length=256, separator=" ",
                 exclude_title=False, add_var_token=False):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.exclude_title = exclude_title
        self.add_var_token = add_var_token

    def __call__(self, example):
        query_text = example["query"]
        eval_meta = {"query_id": example["query_id"], "positive_passages": [], "negative_passages": []}

        if self.add_var_token:
            query_text = VAR_PREFIX + self.separator + example["query"]
        query = self.tokenizer.encode(
            query_text, add_special_tokens=False, max_length=self.query_max_length, truncation=True
        )
        positives = []
        for pos in example["positive_passages"]:
            if self.exclude_title:
                text = pos["text"]
            else:
                text = pos["title"] + self.separator + pos["text"] if "title" in pos else pos["text"]

            if self.add_var_token:
                text = VAR_PREFIX + self.separator + text

            positives.append(
                self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
            )
            eval_meta["positive_passages"].append(pos["docid"])
        negatives = []
        for neg in example["negative_passages"]:
            if self.exclude_title:
                text = neg["text"]
            else:
                text = neg["title"] + self.separator + neg["text"] if "title" in neg else neg["text"]

            if self.add_var_token:
                text = VAR_PREFIX + self.separator + text

            negatives.append(
                self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
            )
            eval_meta["negative_passages"].append(neg["docid"])

        return {"query": query, "positives": positives, "negatives": negatives, "eval_meta": eval_meta}


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, add_var_token=False, separator=" "):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.add_var_token = add_var_token
        self.separator = separator

    def __call__(self, example):
        query_id = example["query_id"]
        text = example["query"]
        if self.add_var_token:
            text = VAR_PREFIX + self.separator + text

        query = self.tokenizer.encode(
            text, add_special_tokens=False, max_length=self.query_max_length, truncation=True
        )
        return {"text_id": query_id, "text": query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=" ", exclude_title=False, add_var_token=False):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator
        self.exclude_title = exclude_title
        self.add_var_token = add_var_token

    def __call__(self, example):
        docid = example["docid"]
        if self.exclude_title:
            text = example["text"]
        else:
            text = example["title"] + self.separator + example["text"] if "title" in example else example["text"]

        if self.add_var_token:
            text = VAR_PREFIX + self.separator + text

        text = self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
        return {"text_id": docid, "text": text}
