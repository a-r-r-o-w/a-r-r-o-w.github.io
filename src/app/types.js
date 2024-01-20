export class ProjectEntity {
  constructor(data) {
    this.url = data.url || null;
    this.name = data.name || '';
    this.description = data.description || '';
    this.languages = data.languages || [];
    this.image = data.image || null;
    this.video = data.video || null;
  }
};

export class PaperImplementationEntity {
  constructor(data) {
    this.url = data.url || null;
    this.name = data.name || '';
    this.description = data.description || '';
    this.arxiv = data.arxiv || null;
    this.colab = data.colab || null;
  }
};
