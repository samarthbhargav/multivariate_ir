class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=" ", include_title=True):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.include_title = include_title

    def __call__(self, example):
        query = self.tokenizer.encode(
            example["query"], add_special_tokens=False, max_length=self.query_max_length, truncation=True
        )
        positives = []
        for pos in example["positive_passages"]:
            if self.include_title:
                text = pos["title"] + self.separator + pos["text"] if "title" in pos else pos["text"]
            else:
                text = pos["text"]
            positives.append(
                self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
            )
        negatives = []
        for neg in example["negative_passages"]:
            if self.include_title:
                text = neg["title"] + self.separator + neg["text"] if "title" in neg else neg["text"]
            else:
                text = neg["text"]
            negatives.append(
                self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
            )
        return {"query": query, "positives": positives, "negatives": negatives}


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example["query_id"]
        query = self.tokenizer.encode(
            example["query"], add_special_tokens=False, max_length=self.query_max_length, truncation=True
        )
        return {"text_id": query_id, "text": query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=" ", include_title=True):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator
        self.include_title = include_title

    def __call__(self, example):
        docid = example["docid"]
        if self.include_title:
            text = example["title"] + self.separator + example["text"] if "title" in example else example["text"]
        else:
            text = example["text"]

        print(text, example["title"])
        text = self.tokenizer.encode(text, add_special_tokens=False, max_length=self.text_max_length, truncation=True)
        return {"text_id": docid, "text": text}
