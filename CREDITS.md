# Credits and third-party notices

This project is a personal study. It uses a public dataset and a public book,
both published by the Brazilian Jurimetrics Association (ABJ) under the MIT
licence. This project is not affiliated with, endorsed by, or sponsored by ABJ.

## Book

> Feliz, Ricardo; Trecenti, Julio. **Metodologia de Pesquisa Jurimétrica**.
> ABJ, 2022. <https://livro.abj.org.br>.
>
> Repository: <https://github.com/abjur/livro>
>
> Licence: MIT. Copyright (c) 2022 ABJ.

The book supplies the research methodology used here: planning, the data
generating process, descriptive statistics, visualisation and the discipline of
separating what is known at filing time from what is only known after a
decision.

## Dataset

> Trecenti, Julio; Hirota, Renata; Witkoski, Katerine; Associação Brasileira de
> Jurimetria. **abjData: Databases Used Routinely by the Brazilian Jurimetrics
> Association**. <https://github.com/abjur/abjData>.
>
> Dataset used: `consumo` (1,000-row sample of TJSP second-degree consumer
> cases).
>
> Licence: MIT. Copyright (c) 2016 Brazilian Jurimetrics Association.

The exact upstream commit and file checksum are recorded in
`src/claimvalue/data/book/SOURCE.json`. The prepared table is generated at
runtime under `RAW_DATA_DIR/book/`, which is gitignored, and is never committed.
It pseudonymises the public case number and removes it; no names, CPFs, CNPJs or
free text are included.

## MIT licence text

The following notice covers the book and the `abjData` package:

```text
MIT License

Copyright (c) 2022 ABJ
Copyright (c) 2016 Brazilian Jurimetrics Association

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Synthetic test data

`src/claimvalue/synthetic.py` generates fictional records for the automated test
suite only. Those records are not used in the published results and are never
committed as data files.
