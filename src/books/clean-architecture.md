# Clean Architecture

[Book URL](https://www.amazon.co.jp/Clean-Architecture-%E9%81%94%E4%BA%BA%E3%81%AB%E5%AD%A6%E3%81%B6%E3%82%BD%E3%83%95%E3%83%88%E3%82%A6%E3%82%A7%E3%82%A2%E3%81%AE%E6%A7%8B%E9%80%A0%E3%81%A8%E8%A8%AD%E8%A8%88-Robert-C-Martin/dp/4048930656)

[実践クリーンアーキテクチャ by nrs](https://nrslib.com/clean-architecture/)

## 概要

## 感想

クリーンアーキテクチャの目的は、「ビジネスロジックが UI、DB、外部 API、フレームワークに依存しないようにすることで柔軟なアプリケーションを構築すること 」である。
技術負債が生じにくい設計ともいえる。本書では MVC フレームワークに近いアプリケーションを想定していると考えられる（同心円の図）。現在のバックエンドとフロントエンドを分ける実装に落とし込むためには工夫が必要そうだ。
Interface を用いて依存関係を適切に管理することで、これを実現している。
