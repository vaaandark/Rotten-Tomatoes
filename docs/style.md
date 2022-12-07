# 代码风格规范

> 基本按照 [Google Python 风格规范](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)

## 缩进
- 使用四个空格来缩进代码，绝对不要使用 tab 更不要 tab 与空格混用

## 空格
- 二元操作符两边加空格，括号内例外，括号内不要有空格
- 不要在逗号, 分号, 冒号前面加空格, 但应该在它们后面加（除了在行尾）
- 参数列表, 索引或切片的左括号前不应加空格

## 文件
使用文件操作结束后显式地关闭它，建议使用`with`语句：
```Python
with open("hello.txt") as hello_file:
    for line in hello_file:
        print line
```

## 注释
本项目使用中文注释，请使用`UTF-8`编码

### 文档字符串、类型注释
以下一段示例包含了文档字符串、类型注释：

```Python
def fetch_smalltable_rows(table_handle: smalltable.Table,
                        keys: Sequence[Union[bytes, str]],
                        require_all_keys: bool = False,
) -> Mapping[bytes, Tuple[str]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
        row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: Optional; If require_all_keys is True only
        rows with values set for all keys will be returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
        b'Zim': ('Irk', 'Invader'),
        b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

重要函数的注释中需要包含对参数、返回值、异常的说明，主要说明这个函数做了什么，不必过于详细。

### 块注释和行注释
最需要写注释的是代码中那些技巧性的部分。如果你在下次**代码审查**的时候必须解释一下，那么你应该现在就给它写注释。对于复杂的操作，应该在其操作开始前写上若干行注释。对于不是一目了然的代码，应在其行尾添加注释。

为了提高可读性，注释应该至少离开代码 2 个空格。

另一方面，绝不要描述代码。假设阅读代码的人比你更懂 Python ，他只是不知道你的代码要做什么。

### TODO 注释
为临时代码使用TODO注释：
```Python
# TODO(kl@gmail.com): Use a "*" here for string repetition.
# TODO(Zeke) Change this to use relations.
```

## 命名
- 模块名写法:`module_name`
- 包名写法:`package_name`
- 类名:`ClassName`
- 方法名:`method_name`
- 异常名:`ExceptionName`
- 函数名:`function_name`
- 全局常量名:`GLOBAL_CONSTANT_NAME`
- 全局变量名:`global_var_name`
- 实例名:`instance_var_name`
- 函数参数名:`function_parameter_name`
- 局部变量名:`local_var_name`

函数名，变量名和文件名应该是描述性的，尽量避免缩写，特别要避免使用非项目人员不清楚难以理解的缩写，不要通过删除单词中的字母来进行缩写。始终使用`.py`作为文件后缀名，不要用连接号。

## Main

```Python
def main():
    ...

if __name__ == '__main__':
    main()
```
