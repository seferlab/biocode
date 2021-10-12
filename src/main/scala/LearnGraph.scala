import ec.Evolve

// For command line parsing
import org.rogach.scallop._

object LearnGraph {

  def main( args: Array[String] ) {
       /**
       * Configuration Options
       */
      object Conf extends ScallopConf(args) {
	    val logo =
	"""
	"""

        version("1.0")
        banner("""Usage: LearnGraph [OPTION]
                  |
                  | """+logo+"""
                  | Learn a graph
                  |
                  |Options:
                  |""".stripMargin)

        val configFile = opt[String]("cfg", 
          descr = "file containing the configuration", required = true )
        try {
          verify
        } catch {
          case v: exceptions.ValidationFailure => println("Error parsing arguments; exiting"); sys.exit()
        }
      }


    Evolve.main(Array("-file", Conf.configFile()))//"src/test/resources/growCodeTest.params"))
  }

}
