from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'artifact'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#163044','axes.labelcolor':'#163044'})
fig=plt.figure(figsize=(16,9),facecolor='#f5f8fc')
fig.text(.06,.91,'Modificar pesos. Medir el resultado.',fontsize=30,weight='bold')
fig.text(.06,.85,'CoSTI · imputación de series temporales · edición sin entrenamiento por gradientes',fontsize=16)
for x,big,small in [(.06,'965','parámetros modificados'),(.38,'0,04010 %','de los parámetros registrados'),(.71,'2.406.657','parámetros registrados')]:
 fig.text(x,.70,big,fontsize=34,weight='bold',color='#007f79');fig.text(x,.65,small,fontsize=14)
for x,title,a,b,pct in [(.06,'MAE',2.480960,2.417163,'−2,5714 %'),(.54,'MSE',350.120368,347.977979,'−0,6119 %')]:
 ax=fig.add_axes([x,.25,.38,.27]);ax.barh(['V4','Original'],[b,a],color=['#008b83','#8c9bad']);ax.set_xlim(0,a*1.12);ax.set_title(title+'  '+pct,loc='left',fontsize=22,weight='bold');ax.spines[['top','right','left']].set_visible(False)
 for y,v in enumerate([b,a]):ax.text(v+.01*a,y,f'{v:.6f}'.replace('.',','),va='center',fontsize=13)
fig.text(.06,.15,'4.033 pacientes · 100 imputaciones · 2 pasos · 2 semillas de ruido del mismo checkpoint',fontsize=15)
fig.text(.06,.09,'Referencia: ejecución propia de los pesos originales. Benchmark reutilizado; no es la media del paper.',fontsize=12)
fig.text(.06,.045,'El 89,74 % de la reducción de MAE procede de variables de contexto. No todas las variables mejoran.',fontsize=12)
for e in ['png','svg','pdf']:fig.savefig(P/f'social_resultados.{e}',dpi=150,facecolor=fig.get_facecolor())
plt.close(fig)
fig=plt.figure(figsize=(16,9),facecolor='#f5f8fc')
fig.text(.06,.91,'De inspeccionar neuronas a programar una corrección',fontsize=27,weight='bold')
fig.text(.06,.84,'Representación interna → cabeza de salida: 64 → 256 (GELU) → 64 → 1',fontsize=19)
for x,t,s in [(.06,'1 · Inspeccionar','Pesos y activaciones\nContribución efectiva\nRegiones de error'),(.38,'2 · Programar','d = z₃₈ − z₁₉₈\nCorrección ≈ 0,0696 · clip(d, 0, 1)\nUnidades reutilizadas: 181 y 150'),(.72,'3 · Comprobar','Forward passes completos\nCalibración y confirmación\nTest y exportación verificada')]:
 fig.text(x,.71,t,fontsize=23,weight='bold',color='#007f79');fig.text(x,.57,s,fontsize=16,linespacing=1.8)
fig.text(.06,.37,'Lo que sabemos',fontsize=20,weight='bold');fig.text(.06,.30,'El conjunto V4 mejora MAE y MSE. La fórmula describe una corrección directa en la cabeza.',fontsize=16)
fig.text(.06,.19,'Lo que falta demostrar',fontsize=20,weight='bold');fig.text(.06,.12,'Generalización externa, efecto aislado de cada componente y utilidad clínica.\nLos dos pasos pueden amplificar o cambiar una corrección local. No se demostró escapar de un mínimo.',fontsize=15,linespacing=1.5)
for e in ['png','svg','pdf']:fig.savefig(P/f'social_mecanismo.{e}',dpi=150,facecolor=fig.get_facecolor())
