class ReportGenerator:
    def __init__(self, history, df):
        self.history = history
        self.df = df

    def generate_text_report(self):
        report = []
        report.append("ML DATA EXPLORER REPORT\n")
        report.append("=" * 40 + "\n\n")

        report.append("Operations Performed:\n")
        for i, h in enumerate(self.history, 1):
            report.append(f"{i}. {h}\n")

        report.append("\nFinal Dataset Shape:\n")
        report.append(str(self.df.shape))

        return "".join(report)
