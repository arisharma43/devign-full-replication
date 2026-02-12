/* graph-for-funcs.scala - Simplified for Joern 4.x compatibility

   Returns JSON representation of methods with AST, CFG, and PDG information
 */

import upickle.default.{ReadWriter, macroRW, write}
import scala.jdk.CollectionConverters._
import io.shiftleft.codepropertygraph.generated.nodes

case class NodeInfo(id: String, label: String, properties: Map[String, String])
case class EdgeInfo(src: String, dst: String, edgeType: String)
case class MethodGraph(function: String, file: String, id: String, nodes: List[NodeInfo], edges: List[EdgeInfo])
case class Result(functions: List[MethodGraph])

given ReadWriter[NodeInfo] = macroRW
given ReadWriter[EdgeInfo] = macroRW
given ReadWriter[MethodGraph] = macroRW
given ReadWriter[Result] = macroRW

def generateGraphs(): String = {
  val methods = cpg.method.l.map { method =>
    val methodName = method.fullName
    val methodId = method.id.toString
    val methodFile = try { method.location.filename } catch { case _: Exception => "N/A" }
    
    // Get all AST nodes
    val astNodes = method.ast.l
    
    // Create node info
    val nodes = astNodes.map { node =>
      val props = node.propertiesMap.asScala.map { case (k, v) => 
        (k, v.toString)
      }.toMap
      NodeInfo(node.id.toString, node.label, props)
    }
    
    // Create edge info (AST and CFG edges)
    val edges = astNodes.flatMap { node =>
      // AST edges
      val astEdges = node.astChildren.map { child =>
        EdgeInfo(node.id.toString, child.id.toString, "AST")
      }
      // CFG edges
      val cfgEdges = try {
        node.asInstanceOf[io.shiftleft.codepropertygraph.generated.nodes.CfgNode]
          .cfgNext.map { next =>
            EdgeInfo(node.id.toString, next.id.toString, "CFG")
          }
      } catch {
        case _: Exception => List.empty
      }
      astEdges.toList ++ cfgEdges.toList
    }.distinct
    
    MethodGraph(methodName, methodFile, methodId, nodes.toList, edges)
  }
  
  val result = Result(methods)
  write(result)
}
