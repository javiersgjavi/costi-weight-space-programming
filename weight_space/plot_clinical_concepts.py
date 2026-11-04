"""Render public aggregate concept figures; no clinical data or model needed."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtr

def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.report.read_text());a.output.mkdir(exist_ok=True,parents=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#17334a','axes.labelcolor':'#17334a'})
 def save(fig,name):
  for ext in ['png','pdf','svg']:fig.savefig(a.output/(name+'.'+ext),dpi=160,facecolor=fig.get_facecolor(),bbox_inches='tight',pad_inches=.25)
  plt.close(fig)
 def heat(ax,m,x,y):
  im=ax.imshow(m,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(len(x)),x,fontsize=10);ax.set_yticks(range(len(y)),y)
  for i in range(m.shape[0]):
   for j in range(m.shape[1]):ax.text(j,i,f'{m[i,j]:+.2f}',ha='center',va='center',color='white' if abs(m[i,j])>.55 else '#17334a',fontsize=13)
  return im
 fig,axes=plt.subplots(1,2,figsize=(16,7),facecolor='#f5f8fc');fig.subplots_adjust(left=.08,right=.91,bottom=.23,top=.77,wspace=.3)
 x=['Nivel\ninterpolado','Pendiente','Variación\ntemporal','Densidad\nobservada','Distancia a\nobservación','Hora']
 for ax,v in zip(axes,['V1','V4']):
  im=heat(ax,np.array(d['confirmation'][v]['matrix']),x,[str(j) for j in d['units']]);ax.set_title(v+' · preactivación antes de GELU',pad=15);ax.set_ylabel('Unidad')
 fig.colorbar(im,cax=fig.add_axes([.935,.23,.014,.54]),label='Correlación de rangos dentro de cada variable')
 fig.suptitle('47 y 255: asociación con el nivel de la señal, replicada en V1 y V4',fontsize=22,y=.94,weight='bold')
 fig.text(.08,.10,'Mismos 32 pacientes · 3.569 entradas evaluables · 100 muestras · último paso de la muestra que determina la mediana',fontsize=12)
 fig.text(.08,.045,'Análisis descriptivo, sin intervalos: la correlación no identifica un concepto clínico ni descarta relaciones no lineales.',fontsize=12)
 save(fig,'conceptos_v1_v4')
 fig,axes=plt.subplots(1,2,figsize=(15,7),facecolor='#f5f8fc');fig.subplots_adjust(left=.11,right=.91,bottom=.22,top=.77,wspace=.3)
 for ax,v in zip(axes,['V1','V4']):
  rows=[r for r in d['input_interventions'][v] if r['step']==2];m=np.array([r['mean_delta_preactivation'] for r in rows])
  im=ax.imshow(m,vmin=-1.2,vmax=1.2,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(4),[str(j) for j in d['units']]);ax.set_yticks(range(4),[r['variable'] for r in rows]);ax.set_title(v+' · cambio medio en preactivación');ax.set_xlabel('Unidad')
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{m[i,j]:+.3f}',ha='center',va='center',color='white' if abs(m[i,j])>.65 else '#17334a',fontsize=14)
 fig.colorbar(im,cax=fig.add_axes([.935,.22,.014,.55]),label='Δ preactivación, último paso')
 fig.suptitle('Qué ocurre al aumentar las observaciones de una variable',fontsize=23,y=.94,weight='bold')
 fig.text(.11,.10,'+0,5 desviaciones estándar de las observaciones de entrenamiento · 2 pacientes por variable · mismo ruido',fontsize=12)
 fig.text(.11,.045,'Media sobre 100 muestras, sin seleccionar una mediana. Sensibilidad matemática en 8 casos; no intervención clínica.',fontsize=12)
 save(fig,'intervenciones_conceptos')
 fig,ax=plt.subplots(figsize=(12,7),facecolor='#f5f8fc');fig.subplots_adjust(left=.12,right=.95,bottom=.23,top=.78)
 u=[38,198];pos=d['calibration']['V4']['positive_regions'];x=np.arange(2)
 aa=np.array([pos[str(j)]['fraction_target_above_2_normalized'] for j in u])*100;bb=np.array([pos[str(j)]['fraction_target_above_95th_observed_training_percentile'] for j in u])*100
 for shift,vals,col,label in [(-.18,aa,'#008b83','Valor normalizado > 2'),(.18,bb,'#e38a37','Valor por encima del percentil 95 observado')]:
  ax.bar(x+shift,vals,.32,color=col,label=label)
  for xx,val in zip(x+shift,vals):ax.text(xx,val+2,f'{val:.1f} %',ha='center',fontsize=15)
 ax.set_xticks(x,['Unidad 38','Unidad 198']);ax.set_ylim(0,115);ax.set_ylabel('% de casos con preactivación positiva');ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.12),ncol=1,frameon=False)
 fig.suptitle('Un valor alto en la escala interna puede ser habitual en la variable',fontsize=19,y=.96,weight='bold')
 fig.text(.12,.11,'V4 · calibración de 512 pacientes · percentiles calculados solo con valores realmente observados en entrenamiento',fontsize=10)
 fig.text(.12,.055,'El escalador original incluye ceros de datos ausentes. El percentil 95 es una referencia estadística, no clínica.',fontsize=11)
 save(fig,'escala_interna_y_extremos')
 fig,axes=plt.subplots(2,2,figsize=(14,9),facecolor='#f5f8fc');fig.subplots_adjust(left=.08,right=.96,bottom=.14,top=.86,wspace=.25,hspace=.4)
 gelu=lambda x:x*ndtr(x)
 coef=d['calibration']['V1']['effective_coefficients']
 z=np.linspace(-4,3,1000);a47=coef['47'];a255=coef['255'];k=.04*coef['198']
 plots=[(axes[0,0],z,a47/1.01*gelu(1.01*z)-a47*gelu(z),'47: cambio en la contribución','z de la unidad 47'),(axes[0,1],z,a255*(gelu(z-.01)-gelu(z)),'255: desplazamiento del sesgo','z de la unidad 255'),(axes[1,0],z,k*(gelu(16*z)-4*gelu(16*(z-1.5))+3*gelu(16*(z-2)))/16,'246 / 226 / 179: corrección localizada','z de la unidad 198')]
 dd=np.linspace(-.2,1.4,1000);plots.append((axes[1,1],dd,.06959482431411744*(gelu(64*dd)-gelu(64*(dd-1)))/64,'181 / 150: contraste con saturación','d = z38 − z198'))
 for ax,x,y,title,xlabel in plots:ax.plot(x,y,color='#008b83',lw=2.5);ax.axhline(0,color='#8898a9',lw=.7);ax.set_title(title,fontsize=14);ax.set_xlabel(xlabel);ax.set_ylabel('Δ salida de la cabeza');ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Funciones explícitas programadas en V4',fontsize=24,y=.96,weight='bold')
 fig.text(.08,.045,'Representación de entrada fija. Son curvas de las ediciones en la cabeza; no el efecto completo tras los dos pasos.',fontsize=12)
 save(fig,'funciones_neuronas_v4')
if __name__=='__main__':main()
