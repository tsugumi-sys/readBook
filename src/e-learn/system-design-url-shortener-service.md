# System Design of Scaleble URL shortener service

source: https://medium.com/@sandeep4.verma/system-design-scalable-url-shortener-service-like-tinyurl-106f30f23a82


## 自分の考え

### システムが設計すべき事項・解決すべき問題・設計スコープ

要件
- 与えられたURLに対し、自分のドメインのURLを生成して割り当てる。
- すでに割り当て済みのURLがあった場合には、生成済みのURLを返す。
- 生成したURLにアクセスがあった場合に、登録された元のURLに対してリダイレクトする。
- 一定期間アクセスがなかったURLは自動で削除する。

### HLD

```mermaid
flowchart TD
    db --> apiServer
    batchServer --> db
    cacheServer --> apiServer
```

### 細部

#### Scaling

- ユーザー数・CPU使用割合に応じたAPIサーバーのスケーリング。
- データベースの量はそこまでいらない。ユーザー当たり、頻繁に短いURLを作成するケースは少ないと想定できるため。

#### Cacheing

- 直近アクセスがあったURLはキャッシュしておく。
