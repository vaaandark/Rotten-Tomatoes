# Git Commit 规范

> 参考博客 [Commit message 和 Change log 编写指南](https://www.ruanyifeng.com/blog/2016/01/commit_message_change_log.html)

本项目采用英文 commit ， commit 只需要出现 `Header.{type, subject}` 和 `Body` 。

## Commit message 的格式
每次提交，Commit message 都包括三个部分：Header，Body 和 Footer。

### Header
Header部分只有一行，包括三个字段：type（必需）、scope（可选）和subject（必需）。

#### type
- feat：新功能（feature）
- fix：修补bug
- docs：文档（documentation）
- style： 格式（不影响代码运行的变动）
- refactor：重构（即不是新增功能，也不是修改bug的代码变动）
- test：增加测试
- chore：构建过程或辅助工具的变动

#### scope
...

#### subject
subject是 commit 目的的简短描述，不超过50个字符。
- 以动词开头，使用第一人称现在时，比如change，而不是changed或changes
- 第一个字母小写
- 结尾不加句号（.）

#### Body
Body 部分是对本次 commit 的详细描述，可以分成多行。下面是一个范例：
```
More detailed explanatory text, if necessary.  Wrap it to 
about 72 characters or so. 

Further paragraphs come after blank lines.

- Bullet points are okay, too
- Use a hanging indent
```

### Footer
...
