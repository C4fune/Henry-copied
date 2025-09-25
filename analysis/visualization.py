import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class AnalyticsVisualizer:
    def __init__(self):
        sns.set_style('whitegrid')
        sns.set_palette('Set2')
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        import os
        self.images_dir = 'images'
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
    
    def _get_image_path(self, filename: str) -> str:
        import os
        if not filename.startswith(self.images_dir):
            filename = os.path.join(self.images_dir, os.path.basename(filename))
        return filename
    
    def create_bar_chart(self, data: pd.DataFrame, title: str, filename: str, **kwargs):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x_col = kwargs.get('x', data.columns[0])
        y_col = kwargs.get('y', data.columns[1])
        
        if len(data) > 20:
            data = data.head(20)
        
        ax.bar(range(len(data)), data[y_col].values, color='steelblue')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data[x_col].values, rotation=45, ha='right')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        
        for i, v in enumerate(data[y_col].values):
            ax.text(i, v, f'{v:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self._get_image_path(filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_heatmap(self, data: pd.DataFrame, title: str, filename: str, **kwargs):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if data.shape[1] > 20:
            data = data.iloc[:, :20]
        if data.shape[0] > 20:
            data = data.iloc[:20, :]
        
        sns.heatmap(data, annot=kwargs.get('annot', True), 
                    fmt=kwargs.get('fmt', '.2f'),
                    cmap=kwargs.get('cmap', 'coolwarm'),
                    center=kwargs.get('center', 0),
                    ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self._get_image_path(filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_line_chart(self, data: pd.DataFrame, title: str, filename: str, **kwargs):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x_col = kwargs.get('x', data.columns[0])
        y_cols = kwargs.get('y', [data.columns[1]])
        
        if not isinstance(y_cols, list):
            y_cols = [y_cols]
        
        for y_col in y_cols:
            ax.plot(data[x_col], data[y_col], marker='o', label=y_col, linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col)
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self._get_image_path(filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_scatter_plot(self, data: pd.DataFrame, title: str, filename: str, **kwargs):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x_col = kwargs.get('x', data.columns[0])
        y_col = kwargs.get('y', data.columns[1])
        
        scatter = ax.scatter(data[x_col], data[y_col], 
                            c=kwargs.get('color', 'steelblue'),
                            s=kwargs.get('size', 50),
                            alpha=0.6)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        
        if kwargs.get('trendline'):
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self._get_image_path(filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_pie_chart(self, data: pd.DataFrame, title: str, filename: str, **kwargs):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        values_col = kwargs.get('values', data.columns[-1])
        labels_col = kwargs.get('labels', data.columns[0])
        
        if len(data) > 10:
            other_sum = data[values_col].iloc[10:].sum()
            data = data.head(10).copy()
            if other_sum > 0:
                data = pd.concat([data, pd.DataFrame({
                    labels_col: ['Others'],
                    values_col: [other_sum]
                })], ignore_index=True)
        
        colors = sns.color_palette('Set3', len(data))
        wedges, texts, autotexts = ax.pie(data[values_col], 
                                           labels=data[labels_col],
            autopct='%1.1f%%',
                                           colors=colors,
                                           startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self._get_image_path(filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_stacked_bar(self, data: pd.DataFrame, title: str, filename: str, **kwargs):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x_col = kwargs.get('x', data.columns[0])
        y_cols = kwargs.get('y', data.columns[1:])
        
        data_to_plot = data.set_index(x_col)[y_cols]
        data_to_plot.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(x_col)
        ax.set_ylabel('Value')
        ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self._get_image_path(filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_visualization(self, viz_type: str, data: pd.DataFrame, 
                             title: str, filename: str, **kwargs):
        filename = self._get_image_path(filename)
        
        viz_methods = {
            'bar': self.create_bar_chart,
            'bar_chart': self.create_bar_chart,
            'heatmap': self.create_heatmap,
            'line': self.create_line_chart,
            'line_chart': self.create_line_chart,
            'scatter': self.create_scatter_plot,
            'scatter_plot': self.create_scatter_plot,
            'pie': self.create_pie_chart,
            'pie_chart': self.create_pie_chart,
            'stacked_bar': self.create_stacked_bar
        }
        
        viz_method = viz_methods.get(viz_type, self.create_bar_chart)
        viz_method(data, title, filename, **kwargs)

