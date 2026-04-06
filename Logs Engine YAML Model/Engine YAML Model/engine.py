from pydantic import BaseModel, Validator
from typing import List, Dict, Any
import yaml

class RuleModel(BaseModel):
    rule_id: str
    priority: int
    condition: str
    actions: List[Dict[str, Any]]

    @validator("rule_id","condition")
    def not_empty(cls,v):
        if not v.strip():
            raise ValueError("Campo obrigatório.")
        return v
    
    def load_rules(path:str):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            rules = [RuleModel(**item) for item in raw]
        return rules

Class RulesEngine():
    def __init__(self, rules:List[RuleModel]):
        self.rules[for r in rules]
    
    def evaluate(self, value:float):
        print (f"Avaliar o valor:{value}")
        for r in self.rules:
            if eval(r.condition):
                print (f"\n Regra ativa:{r.rule.id}")
                self.execute_actions(r.actions)
        
    def execute_actions(self, actions):
        for actions in action:
            action_type = next(iter(action))
            params = action[action_type]

        if action_type == "send_alert":
            print (f"[ALERTA] {params['message']}")

        elif action_type == "log_event":
            print (f"[LOG] {params['level']}")
    
