# Third-party notices

The adapters use upstream forecasting implementations. Repository URLs, exact commits, and recorded redistribution status are preserved in [model_sources.lock.yaml](code/third_party/model_sources.lock.yaml); required file hashes are recorded in [source_hashes.json](code/third_party/source_hashes.json).

The following table summarizes that local source lock, captured on 2026-09-13. It is not a claim about subsequent upstream licensing changes.

| Backbone | Upstream repository | License recorded in the source lock |
| --- | --- | --- |
| iTransformer | [thuml/iTransformer](https://github.com/thuml/iTransformer) | MIT |
| TimeMixer | [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) | Apache-2.0 |
| ModernTCN | [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) | MIT |
| SOFTS | [Secilia-Cxy/SOFTS](https://github.com/Secilia-Cxy/SOFTS) | MIT |
| TimeXer | [thuml/TimeXer](https://github.com/thuml/TimeXer) | No root license file recorded; redistribution withheld |
| TimeBridge | [Hank0626/TimeBridge](https://github.com/Hank0626/TimeBridge) | MIT |
| TimeFilter | [TROUBADOUR000/TimeFilter](https://github.com/TROUBADOUR000/TimeFilter) | No root license file recorded; redistribution withheld |
| SRSNet | [decisionintelligence/SRSNet](https://github.com/decisionintelligence/SRSNet) | MIT |
| ChannelTokenFormer | [jinkwan1115/ChannelTokenFormer](https://github.com/jinkwan1115/ChannelTokenFormer) | No root license file recorded; redistribution withheld |

## Installation and distribution

`code/setup_sources.py` fetches only the required files at locked revisions and verifies their hashes. Downloaded sources live in `code/third_party/sources/`, which is excluded from Git. Their licenses and notices must be retained when redistribution is permitted.

The local preparation copy includes the six permitted minimal source subsets with their license files. It excludes TimeXer, TimeFilter, and ChannelTokenFormer upstream source files; these are obtained from their upstream repositories during setup. Downloading a component does not itself grant a redistribution license. Any separately prepared archive must preserve these exclusions and applicable notices.

STR and shared MEF components are documented in [implementation notes](code/IMPLEMENTATION.md), with provenance in [SOURCE_MANIFEST.json](code/SOURCE_MANIFEST.json). The project's own license has not yet been selected. No blanket license is applied to third-party components or the accompanying datasets.
