"""Pydantic v2 models matching the ConvFinQA dataset schema."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    pre_text: str = Field(description="Text before the table")
    post_text: str = Field(description="Text after the table")
    table: dict[str, dict[str, float | str | int]] = Field(description="Table as year -> metric -> value")


class Dialogue(BaseModel):
    conv_questions: list[str] = Field(description="Questions in the conversation")
    conv_answers: list[str] = Field(description="Human-readable answers")
    turn_program: list[str] = Field(description="DSL program for each turn")
    executed_answers: list[float | str] = Field(description="Gold numeric answers")
    qa_split: list[bool] = Field(description="Source question split (False=first, True=second FinQA question)")


class Features(BaseModel):
    num_dialogue_turns: int = Field(description="Number of turns in the dialogue")
    has_type2_question: bool = Field(description="Whether dialogue has a Type II hybrid question")
    has_duplicate_columns: bool = Field(description="Whether table has duplicate column headers")
    has_non_numeric_values: bool = Field(description="Whether table has non-numeric values")


class ConvFinQARecord(BaseModel):
    id: str = Field(description="Unique record identifier")
    doc: Document = Field(description="The financial document")
    dialogue: Dialogue = Field(description="The conversational dialogue")
    features: Features = Field(description="Record metadata")
