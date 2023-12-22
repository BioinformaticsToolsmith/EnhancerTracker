class Enhancer:

    def __init__(self, chrom, start, end, confidence):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.confidence = confidence

    def __str__(self):
        return f"{self.chrom}:{self.start}-{self.end} ({self.confidence})"
    
    def to_bed(self, with_confidence = False):
        return f"{self.chrom}\t{self.start}\t{self.end}\t{self.confidence}" if with_confidence else f"{self.chrom}\t{self.start}\t{self.end}"
    
    def get_chrom(self):
        return self.chrom
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end
    
    def get_confidence(self):
        return self.confidence
    
    def get_length(self):
        return self.end - self.start
    
    

