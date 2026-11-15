# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Delta L: detecção de mudanças de superfície com NISAR + confirmação multissatélite.

Módulos:
    grid       grade comum (UTM) para todas as fontes
    change     contas: dB, filtro, razão logarítmica, z robusto, limiar, área mínima
    layers     três camadas: vegetação/fogo, água, dano/movimento
    validate   matriz de confusão e métricas
    sources_pc     Microsoft Planetary Computer (Sentinel-2, Sentinel-1 RTC, DEM)
    sources_nisar  NISAR via catálogo da NASA (CMR) e earthaccess
    sources_power  chuva diária do NASA POWER
    report     figuras, ficha técnica (JSON/CSV) e texto do resultado

Parte do código foi escrita com apoio de IA (Claude); revisado pela equipe Delta L.
"""
__version__ = "0.1.0"
