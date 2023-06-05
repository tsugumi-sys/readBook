# System Design Interview Basics

source: https://www.youtube.com/watch?v=Gg318hR5JY0&t=18s

## Cmmunication

大きく分けて、以下の２つを見ている。
- 問題解決能力
- 他者とのコミュニケーション能力

抽象的な課題（システムデザイン）に対して、問題設定から要件の洗い出し・技術選定までのプロセスをどれくらい高い質で行えるかが見られている。

また、他人に相談することで適切なコミュニケーションを行えるかどうかも確認される。

したがって、自分の考えを透明にする（書き起こす・口に出す）ことを重視する。

## Designing to Scale

今日において、どれくらいのアクセス・データ量を捌けるかどうかは必須の要件となっている。

System Propertiesが代表的にな要件となっている。

### System Properties

- Latency
- Throughput
- Storage

## Concrete and Quantitative Solutions

常に物理法則や現実的なパラメータ（距離など）も合わせて考える必要がある。

### Costs of Operations

- Read from disk
- Read from memory
- Local Area Network (LAN) round-trip
- Cross-Continental Network

実際にはホワイトボードにて、概要図（アーキテクチャー図）を使うことが多い。
これらの課題を解決するための代表的なパターンがいくつかあるため、押さえておこう。

### Solution Patterns

- Sharding data
- Replication types
- Write-Ahead Logging
- Separating data and metadata storage
- Load distribution

## Trade-offs and Compromises

実際にシステムデザインを行うときには、いくつかのトレードオフに基づいて妥協しなければならない。
インタビュー時にこれらのことも議論できるとなお良い。