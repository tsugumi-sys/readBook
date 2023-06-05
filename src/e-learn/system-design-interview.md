# System Design Interview Basics

## はじめに

Googleがシステムデザインインタビューに関する動画で基礎中の基礎を書いてくれていたので、まとめる。

source: https://www.youtube.com/watch?v=Gg318hR5JY0&t=18s

### Cmmunication

大きく分けて、以下の２つを見ている。

- 問題解決能力
- 他者とのコミュニケーション能力

#### 問題解決能力

技術力関係なしに、その人の問題解決能力がどれくらいあるかをまず測りたい。
したがって、自分の考えたことを相手にも積極的に伝えて透明化することを意識する。

加えて、技術的な知識・経験等を組み合わせてどれくらい質の高いアウトプットが出せるかをみたい。


#### コミュニーケーション能力

議論をリードする力、問題を構造化して一つづつ確実に処理していく整理力、詰まったところなどを整理して他者と相談しがなら進められる力など。

### Designing to Scale

今日において、どれくらいのアクセス・データ量を捌けるかどうかは必須の要件となっている。

System Propertiesが代表的にな要件となっている。

#### System Properties

- Latency
- Throughput
- Storage

https://www.youtube.com/watch?v=-W9F__D3oY4&t=2s

### Concrete and Quantitative Solutions

常に物理法則や現実的なパラメータ（距離など）も合わせて考える必要がある。

#### Costs of Operations

- Read from disk
- Read from memory
- Local Area Network (LAN) round-trip
- Cross-Continental Network

実際にはホワイトボードにて、概要図（アーキテクチャー図）を使うことが多い。
これらの課題を解決するための代表的なパターンがいくつかあるため、押さえておこう。

#### Solution Patterns

- Sharding data
- Replication types
- Write-Ahead Logging
- Separating data and metadata storage
- Load distribution

### Trade-offs and Compromises

実際にシステムデザインを行うときには、いくつかのトレードオフに基づいて妥協しなければならない。
インタビュー時にこれらのことも議論できるとなお良い。


## 実際の流れ一例

1. そのシステムが達成すべき事項・解決すべき問題の理解・設計スコープの設定
2. high-level architecture の提案
3. より細部の設計検討
4. ラップアップ

「そのシステムを設計する上で重要な課題は何か？」という問題の理解に時間をかけずに、稚速にアーキテクチャや採用すべき技術を提案することは百害あって一利なし。

問題を理解するために、質問を適切に投げかけて仮定・想定を少しづつ積み上げていく。これによって言語化されていなかった要件や仕様を明確にしていく。

### 問題理解のための質問具体例

- 具体的にどのような機能が必要なのか。
- 想定ユーザー数は。
- システムに期待するスケール速度は？
- その企業では今、どんな技術スタックを採用しているか？既存のサービスを流用することで新規の作成範囲を縮小できないか。

### high level architectureについて

システム全体のコンセプトを伝えることを心がある。
また、この段階で性能や収容性・スケーラビリティについても荒く計算し最低限の実現性を担保しておく。

### ラップアップについて

- コンポーネントの故障時の可用性について、どのように考えているのか？
- 運用する上で監視するべきメトリクスは何か？
- ユーザー数増加に伴うシステムのスケーリング戦略はあるか？


## 有用なオンライン学習ソース

- https://github.com/checkcheckzz/system-design-interview
- https://github.com/donnemartin/system-design-primer/blob/master/README-ja.md
- https://www.youtube.com/watch?v=-W9F__D3oY4&t=2s
