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

    def _preproc(self, text, title):
        if not self.exclude_title:
            text = title + self.separator + text if title else text

        if self.add_var_token:
            text = VAR_PREFIX + self.separator + text

        return text

    def __call__(self, example):
        eval_meta = {"query_id": example["query_id"], "positive_passages": [], "negative_passages": []}

        query = self.tokenizer.encode(
            self._preproc(example["query"], None), add_special_tokens=False, max_length=self.query_max_length,
            truncation=True
        )
        positives = []
        for pos in example["positive_passages"]:
            positives.append(
                self.tokenizer.encode(self._preproc(pos["text"], pos["title"]),
                                      add_special_tokens=False,
                                      max_length=self.text_max_length, truncation=True)
            )
            eval_meta["positive_passages"].append(pos["docid"])

        negatives = []
        for neg in example["negative_passages"]:
            negatives.append(
                self.tokenizer.encode(self._preproc(neg["text"], neg["title"]),
                                      add_special_tokens=False,
                                      max_length=self.text_max_length,
                                      truncation=True)
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

    def _preproc(self, text, title):
        if not self.exclude_title:
            text = title + self.separator + text if title else text

        if self.add_var_token:
            text = VAR_PREFIX + self.separator + text

        return text

    def __call__(self, example):
        docid = example["docid"]

        text = self._preproc(example["text"], example["title"])

        text = self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
        return {"text_id": docid, "text": text}
