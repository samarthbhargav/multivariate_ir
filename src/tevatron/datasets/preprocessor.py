class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=" "):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def _preproc(self, example):
        is_variational = len(self.tokenizer.tokenize("[VAR]")) > 0
        if "title" in example.keys():
            text = example["title"] + self.separator + example["text"]
        else:
            text = example["text"]
        if is_variational:
            text = "[VAR]" + self.separator + text
        return text


    def __call__(self, example):
        query = self.tokenizer.encode(
            example["query"], add_special_tokens=False, max_length=self.query_max_length, truncation=True
        )
        positives = []
        for pos in example["positive_passages"]:
            text = self._preproc(pos)
            positives.append(
                self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
            )
        negatives = []
        for neg in example["negative_passages"]:
            text = self._preproc(neg)
            negatives.append(
                self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
            )
        return {"query": query, "positives": positives, "negatives": negatives}


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, separator=" "):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.separator = separator

    def _preproc(self, example):
        is_variational = len(self.tokenizer.tokenize("[VAR]")) > 0
        if is_variational:
            return example["query"]
        return "[VAR]" + self.separator + example["query"]

    def __call__(self, example):
        query_id = example["query_id"]
        text = self._preproc(example)
        query = self.tokenizer.encode(
            text, add_special_tokens=False, max_length=self.query_max_length, truncation=True
        )
        return {"text_id": query_id, "text": query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=" "):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def _preproc(self, example):
        is_variational = len(self.tokenizer.tokenize("[VAR]")) > 0
        if "title" in example.keys():
            text = example["title"] + self.separator + example["text"]
        else:
            text = example["text"]
        if is_variational:
            text = "[VAR]" + self.separator + text
        return text

    def __call__(self, example):
        docid = example["docid"]
        text = self._preproc(example)
        doc = self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
        return {"text_id": docid, "text": doc}
