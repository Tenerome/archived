
package servlets;
import entity.*;
import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

    @WebServlet("/add")
    public class add extends HttpServlet {
        public add() {
            super();
        }

        public void doGet(HttpServletRequest request, HttpServletResponse response)
                throws ServletException, IOException {
            doPost(request, response);
        }

        public void doPost(HttpServletRequest request, HttpServletResponse response)
                throws ServletException, IOException {
            request.setCharacterEncoding("UTF-8");
            response.setContentType("text/html;charset=UTF-8");

            String info = "";
            if (db.add(new student(request))) {
                info = "添加成功";
            } else {
                info = "添加失败";
            }
            request.setAttribute("info", info);
            String url = "index.jsp?operate=output.jsp";
            request.getRequestDispatcher(url).forward(request, response);
        }
    }
