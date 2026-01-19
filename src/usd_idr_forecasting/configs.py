
import yaml
from pydantic import BaseModel
from jinja2 import Template, Environment, FileSystemLoader
import os


class ProjectConfig(BaseModel):
	project_name: str
	wandb_team_name: str
	general: dict
	model: dict
	tuner: dict
	dataset: dict

	@classmethod
	def from_yaml(cls, config_path: str, **kwargs) -> 'ProjectConfig':
		# if rnn_mode not in ['lstm', 'gru']:
		# 	raise ValueError("rnn_mode must be either 'lstm' or 'gru'")
		
		search_path = os.path.dirname(os.path.abspath(config_path)) or '.'
		env = Environment(
			loader=FileSystemLoader(search_path),
			variable_start_string='${{',
            variable_end_string='}}'
		)

		template_name = os.path.basename(config_path)
		template = env.get_template(template_name)
		rendered_config = template.render(**kwargs)
		config_dict = yaml.safe_load(rendered_config)
		
		return cls(**config_dict)