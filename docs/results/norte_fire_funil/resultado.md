# Incêndio Norte, San Clemente e San Juan Capistrano (Califórnia)

Evento de referência: Incêndio Norte (CAL FIRE): ~570 acres, Portico Del Norte e Eminencia Del Norte, San Clemente (2026-10-16).

## Funil de coleta

| Fonte | catálogo | cobre a área | qualidade | pareados | aproveitamento |
|---|---|---|---|---|---|
| Sentinel-2 L2A | 15 | 15 | 12 | 2 | 13.3% |
| Landsat 8/9 Coleção 2 Nível 2 (óptico + térmico) | 15 | 11 | 8 | 2 | 13.3% |
| Sentinel-1 RTC (banda C) | 20 | 20 | 20 | 2 | 10.0% |
| ASTER L1T (térmico 90 m) | 0 | 0 | - | 0 | - |
| Copernicus DEM 30 m | 1 | 1 | - | - | 100.0% |
| NAIP (foto aérea dos EUA, ~0,6 m) | 5 | 2 | - | 1 | 20.0% |
| NISAR GCOV (banda L) | 19 | 16 | 12 | 2 | 10.5% |

## verdade_sentinel2
- area queimada ha: 36.08
- maior grupo ha: 7.56
- area oficial ha: 230.7
- razao maior grupo oficial: 0.0328

## relevo
- declividade media graus: 7.005

## sentinel1
- Perda de vegetação detectada: 116.92 ha; água nova: 0.0 ha.
- intervalo dias: 6.0
- pixels mascarados relevo: 3
- metricas contra a verdade de campo: precisão 0.0144, sensibilidade 0.0466, F1 0.022, IoU 0.0111, falso alarme 0.009.

## landsat
- intervalo dias: 15.0
- lst mediana antes c: 37.1122
- lst mediana depois c: 28.2766
- area dnbr ha: 203.76
- area aquecida ha: 1011.04
- metricas dnbr contra a verdade de campo: precisão 0.0281, sensibilidade 0.2072, F1 0.0495, IoU 0.0254, falso alarme 0.0193.
- metricas termico contra a verdade de campo: precisão 0.0025, sensibilidade 0.0913, F1 0.0049, IoU 0.0024, falso alarme 0.0978.
- aquecimento na cicatriz c: -1.9261

## nisar
- Perda de vegetação detectada: 21.12 ha; água nova: 0.0 ha.
- intervalo dias: 12.0
- metricas contra a verdade de campo: precisão 0.0, sensibilidade 0.0, F1 None, IoU 0.0, falso alarme 0.0016.

## consenso
- area consenso ha: 97.4
- metricas contra a verdade de campo: precisão 0.0103, sensibilidade 0.0277, F1 0.015, IoU 0.0075, falso alarme 0.0075.
- fontes disponiveis media: 3.5968

Chuva entre as datas (NASA POWER): 34.6 mm.

Parte do código e do texto teve apoio de IA (Claude); números calculados pelo pipeline e revisados pela equipe.