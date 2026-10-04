"""iTransformer / MEF-On; model weights are shared with the other setting."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from train import main
if __name__=='__main__':
    sys.argv[1:1]=['--model','iTransformer','--mode','on']
    main()
