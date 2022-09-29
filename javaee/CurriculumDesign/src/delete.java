
package servlets;
import entity.*;
import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;

@WebServlet("/delete")
public class delete extends HttpServlet {
    public delete(){
        super();
    }
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        doPost(request, response);
    }

public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("UTF-8");
        response.setContentType("text/html;charset=UTF-8");

        String info="";
		String id=request.getParameter("id");
		if(db.delete(id)) {
			info="删除成功";
		}else {
			info="删除失败";
		}
		request.setAttribute("info", info);
		String url="index.jsp?operate=output.jsp";
		request.getRequestDispatcher(url).forward(request, response);
}
}
