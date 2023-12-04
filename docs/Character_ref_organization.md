# Organization of the Character Reference Directory

The character reference directory should contain subfolders with character images. Moreover, it can be organized in a hierarchical way following the convention of `character/appearance/outfits/accessories/objects/extras`, as in the following example.


```
.
├── Noise
├── KarenH
├── Mascarail
├── Melakonsi
├── Melca
├── Millicent
├── Sakuna
├── Terakomari
│   ├── cone hair bun
│   │   ├── red dress
│   │   └── uniform
│   ├── hair bun
│   └── none
│       └── pajama
└── Villhaze
```

## Character Name, Appearance, and Character Embedding

For the character level, you can use any name that starts with "Noise" or "noise" to put images of characters or random people that you do not want to get classified as targeted characters. This is useful to prevent from wrong classification results.

For the appearance level, use something that starts with `_` to have a single embedding for character and appearance. For example, putting `_cone hair bun` under `Terakomari` would result in `Terakomari_cone_hair_bun` to be considered as individual embedding when saving embedding initialization information and this is also what will be used in captions. Otherwise, `Terakomari, cone hair bun` is used in captions (`, ` can be replaced by other separators by specifying `--character_inner_sep`) 

You can put `None` or `none` to skip any level so that they are not used in captions. In the above example, the caption would be `Terakomari, pajama` and **not** `Terakomari, none, pajama`. Generally speaking, current character classification mechanism with ccip embeddings work sufficiently well up to this level.


## Outfits and More

Starting from this level it is possible to have multiple items in the folder name, separated by `+`, e.g. `red uniform+black skirt`. Anything starting with `_` will be considered as embeddings (**TODO**: this is to be implemented). When multiple items of a same type exists, they are separated by `--caption_inner_sep` in captions, while different types of items are separated by `--character_outer_sep`. It is important to note that ccip embeddings do not work well for outfits and beyond, so manual inspection will be needed after stage 3 to put everything in order. (Hopefully we get cwip done soon).
