from django import forms

class StockForm(forms.Form):
    symbol = forms.CharField(label='股價代碼', max_length=10)
    start_date = forms.DateField(label='開始日期', widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(label='結束日期', widget=forms.DateInput(attrs={'type': 'date'}))
