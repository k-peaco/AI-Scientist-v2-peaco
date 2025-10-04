import os

from ai_scientist.tools.base_tool import BaseTool


class GetPublicCommentSummaryTool(BaseTool):
    """
    特定のtxtファイル（public_comment_summary.txt）の内容を返すツール。
    パラメータは不要で、ファイルパスはツール内にベタ書きされています。
    """

    def __init__(self):
        super().__init__(
            name="GetPublicCommentSummaryTool",
            description="「次期個人番号カードタスクフォース中間とりまとめ骨子」に対してのパブリックコメントの全文を取得して返します。",
            parameters=[]
        )
        self.file_path = os.path.join(
            os.path.dirname(__file__),
            "../ideas/public_comment_summary.txt"
        )

    def use_tool(self, **kwargs) -> str:
        """
        public_comment_summary.txtファイルの内容を読み込んで返します。
        
        Args:
            **kwargs: 親クラスとの互換性のためのパラメータ（このツールでは使用しません）
            
        Returns:
            str: ファイルの内容、またはエラーメッセージ
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"ファイルの読み込みに失敗しました: {e}"
