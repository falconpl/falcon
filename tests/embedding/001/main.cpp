/*
   FALCON - The Falcon Programming Language.
   FILE: main.cpp

   Example of application provided functions, classes and modules.

   This progam demonstrates how to push functions and classes into
   a script via module injection.

   The application creates a module that exports all the required
   functions and classes, then it injects it in the main module space
   of the target processes before the script is run.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falcon.h>
#include <iostream>


/*
 * Use an anonymous namespace if you don't want to export the functions and
 * classes across C/C++ sources.
 */
namespace {

   /*
    * A simple function; it accepts a number, mandatory, and an optional
    * string, and replicates the string the given number of times.
    *
    * If the string is not given the value " " is assumed.
    */
   FALCON_DECLARE_FUNCTION(replicate, "number:N,string:[S]")
   FALCON_DEFINE_FUNCTION_P1(replicate)
   {
      Falcon::int64 number;
      Falcon::String* string;
      Falcon::String dflt(" ");

      // the FALCON_[N]PCHECK_GET macro helps to check for mandatory  parameters
      if( ! FALCON_NPCHECK_GET(0, Ordinal, number)
          // the O_GET version accepts optional parameters, with a default value if not given
          || ! FALCON_NPCHECK_O_GET(1, String, string, &dflt) )
      {
         // the Function::paramError() method will generate a standardized error description
         // for incongruent paramters.
         throw paramError();
      }

      Falcon::String* value = new Falcon::String(string->replicate(number));

      // This is the correct way to handle a value to the garbage collector
      Falcon::Item retval = FALCON_GC_HANDLE(value);

      // every function must return its frame, or the engine will stay in the frame.
      ctx->returnFrame( retval );
   }


   //===============================================================================
   // Classes
   //===============================================================================

   // This is our structure:
   struct Point {
      int x;
      int y;

      // we might want to have some extra space for GC accounting.
      Falcon::uint32 gcMark;
   };

   // and our handler
   class ClassPoint: public Falcon::Class
   {
   public:
      // Minimal stuff:
      ClassPoint();
      virtual ~ClassPoint();

      // mandatory handlers:
      virtual void dispose( void* instance ) const;
      virtual void* clone( void* instance ) const;
      virtual void* createInstance() const;

      // Relevant Handlers
      virtual void gcMarkInstance( void* instance, Falcon::uint32 mark ) const;
      virtual bool gcCheckInstance( void* instance, Falcon::uint32 mark ) const;

      // Nice to have handlers
      virtual Falcon::int64 occupiedMemory( void* instance ) const;
      virtual void describe( void* instance, Falcon::String& target, int depth = 3, int maxlen = 60 ) const;

      // And... well, you'll want them handlers.
      virtual void store( Falcon::VMContext* ctx, Falcon::DataWriter* stream, void* instance ) const;
      virtual void restore( Falcon::VMContext* ctx, Falcon::DataReader* stream ) const;
   };

   // constructor of our class.
   FALCON_DECLARE_FUNCTION(init, "x:I,y:I")
   FALCON_DEFINE_FUNCTION_P1(init)
   {
      Falcon::int64 x, y;

      if( ! FALCON_NPCHECK_GET(0, Integer, x) || ! FALCON_NPCHECK_GET(1, Integer, y) )
      {
         throw paramError();
      }

      // Configure the structure we're given in ctx->self()
      struct Point* pt = ctx->tself<struct Point>();
      pt->x = (int) x;
      pt->y = (int) y;

      // the constructor should always return the self item.
      ctx->returnFrame(ctx->self());
   }


   // This method shows how to determine the type of an object passed as paramter.
   FALCON_DECLARE_FUNCTION(distance, "pt:Point")
   FALCON_DEFINE_FUNCTION_P1(distance)
   {
      // get somewhere the class we want to check. -- luckily, it's our same class, so...
      static Falcon::Class* pointClass = this->methodOf();
      Falcon::Item* i_pt = ctx->param(0);

      if( i_pt == 0 || ! i_pt->isInstanceOf( pointClass ) )
      {
         throw paramError();
      }

      // we now must be sure to get the correct user data for the point class out of this item.
      struct Point* pt2 = i_pt->castInst<Point>( pointClass );

      // while, we know our self holds a correct pointer
      struct Point* pt1 = ctx->tself<Point>();

      Falcon::numeric d1 = static_cast<Falcon::numeric>(pt1->x - pt2->x);
      Falcon::numeric d2 = static_cast<Falcon::numeric>(pt1->y - pt2->y);

      Falcon::numeric distance = static_cast<Falcon::numeric>(
               sqrt(d1*d1 + d2*d2)
      );

      // flat data do not require GC
      ctx->returnFrame( distance );
   }

   // now, accessor for properties:
   void get_x(const Falcon::Class*, const Falcon::String&, void *instance, Falcon::Item& value )
   {
      struct Point* pt = static_cast<struct Point*>(instance);
      value = static_cast<Falcon::int64>(pt->x);
   }

   void set_x(const Falcon::Class*, const Falcon::String&, void *instance, const Falcon::Item& value )
   {
      // we are to check for value to be something sensible.
      if( ! value.isOrdinal() )
      {
         throw FALCON_SIGN_XERROR(Falcon::ParamError, Falcon::e_inv_params, .extra("N") );
      }

      struct Point* pt = static_cast<struct Point*>(instance);
      pt->x = (int) value.asOrdinal();
   }

   void get_y(const Falcon::Class*, const Falcon::String&, void *instance, Falcon::Item& value )
   {
      struct Point* pt = static_cast<struct Point*>(instance);
      value = static_cast<Falcon::int64>(pt->y);
   }

   void set_y(const Falcon::Class*, const Falcon::String&, void *instance, const Falcon::Item& value )
   {
      // we are to check for value to be something sensible.
      if( ! value.isOrdinal() )
      {
         throw FALCON_SIGN_XERROR(Falcon::ParamError, Falcon::e_inv_params, .extra("N") );
      }

      struct Point* pt = static_cast<struct Point*>(instance);
      pt->y = (int) value.asOrdinal();
   }

   // Let's define our class:
   ClassPoint::ClassPoint():
            Class("Point")
   {
      // the constructor
      setConstuctor( new FALCON_FUNCTION_NAME(init) );

      // a method
      addMethod( new FALCON_FUNCTION_NAME(distance) );

      // and properties:
      addProperty("x", &get_x, &set_x);
      addProperty("y", &get_y, &set_y);

   }

   ClassPoint::~ClassPoint()
   {
      // nothing to delete in class
   }

   // This method is called when the GC thinks an object should be destroyed.
   void ClassPoint::dispose( void* instance ) const
   {
      struct Point* pt = static_cast<struct Point*>(instance);
      delete pt;
   }

   // This method is called when the user asks ofr an explicit cloning.
   void* ClassPoint::clone( void* instance ) const
   {
      struct Point* pt = static_cast<struct Point*>(instance);
      struct Point* copy = new struct Point;

      copy->x = pt->x;
      copy->y = pt->y;
      return copy;
   }

   // This is invoked prior the constructor receives an unconfigured <self> object.
   void* ClassPoint::createInstance() const
   {
      struct Point* pt = new struct Point;
      pt->x = 0;
      pt->y = 0;
      return pt;

      // you can create the item directly in the constructor, if you wish, using:
      // return FALCON_CLASS_CREATE_AT_INIT;
   }

   void ClassPoint::gcMarkInstance( void* instance, Falcon::uint32 mark ) const
   {
      struct Point* pt = static_cast<struct Point*>(instance);
      pt->gcMark = mark;
   }

   // this tells the GC if an object is alive or not.
   bool ClassPoint::gcCheckInstance( void* instance, Falcon::uint32 mark ) const
   {
      struct Point* pt = static_cast<struct Point*>(instance);
      return pt->gcMark >= mark;
   }


   Falcon::int64 ClassPoint::occupiedMemory( void* ) const
   {
      // return the estemed size of the object.
      // We add 16 to account for extra memory blocks used by new allocator.
      return (Falcon::int64) sizeof(struct Point) + 16;
   }

   void ClassPoint::describe( void* instance, Falcon::String& target, int, int) const
   {
      struct Point* pt = static_cast<struct Point*>(instance);
      // we want to render this object to string as "Point (x,y)"
      target = Falcon::String("Point (").N(pt->x).A(",").N(pt->y).A(")");
   }

   // basic serialization step:
   void ClassPoint::store( Falcon::VMContext*, Falcon::DataWriter* stream, void* instance ) const
   {
      struct Point* pt = static_cast<struct Point*>(instance);
      // the static cast is a formality to avoid data mismatch on old compilers where int might not be 32 bit.
      stream->write(static_cast<Falcon::int32>(pt->x));
      stream->write(static_cast<Falcon::int32>(pt->y));
   }

   void ClassPoint::restore( Falcon::VMContext* ctx, Falcon::DataReader* stream ) const
   {
      // the de-serialization requires us to put a fully configured item on top of the context,
      // We also need to store the item in the GC.

      Falcon::int32 x, y;
      // first we shall read the components of the object,
      // so, in case of throws from I/O, we won't leak.

      stream->read(x);
      stream->read(y);

      struct Point* pt = new struct Point;
      pt->x = x;
      pt->y = y;

      // time to give our object back to the engine.
      // As THIS is the class handling the object...
      ctx->pushData( FALCON_GC_STORE(this, pt) );
   }

   //===============================================================================
   // Moduele
   //===============================================================================

   class AppModule: public Falcon::Module
   {
   public:
      AppModule();
      virtual ~AppModule();
   };

   AppModule::AppModule():
            Module("AppModule")
   {
      // all we have to do here is just add the various mantras we want to expose:
      *this
      << new ClassPoint
      << new FALCON_FUNCTION_NAME(replicate)
      ;
   }

   AppModule::~AppModule()
   {}
}


//===============================================================================
// The application
//===============================================================================


int main(int argc, char* argv[] )
{
   Falcon::Engine::init();
   Falcon::VMachine vm;

   std::cout << "The Falcon Programming Language" << std::endl
             << "Embedding test 001 -- Application pushing symbols in scripts." << std::endl
            ;

   if( argc < 2 )
   {
      std::cout << "Usage: 000 <scriptname> <arg1>...<argN>" << std::endl;
   }
   else
   {
      // create a process.
      Falcon::Process* myProcess = vm.createProcess();

      // Create an array containing the parameters for the script.
      Falcon::ItemArray* args = new Falcon::ItemArray;
      for( int i = 2; i < argc; ++i )
      {
         args->append( FALCON_GC_HANDLE(new Falcon::String(argv[i])) );
      }

      // export the symbol as "args"
      Falcon::Item i_args( FALCON_GC_HANDLE(args) );
      // and, why not, the core module as well.
      myProcess->modSpace()->add( Falcon::Engine::instance()->getCore() );

      myProcess->modSpace()->setExportValue("args", i_args );


      // Add our applicaiton module; it's wise to create a new module for each modspace,
      // although, if wished, and properly protected against concurrency,
      // you might share the same module across different ones.
      myProcess->modSpace()->add( new AppModule );

      // let's try to run the script.
      try {
         myProcess->startScript(Falcon::URI(argv[1]), true);
         // wait for the process to complete.
         myProcess->wait();

         Falcon::AutoCString resultDesc( myProcess->result().describe(3, 128) );
         std::cout << "====================================================" << std::endl;
         std::cout << "Script completed with result: " << resultDesc.c_str() << std::endl;
      }
      catch( Falcon::Error* error )
      {
         Falcon::AutoCString desc( error->describe(true,true,false) );
         std::cout << "FATAL: Script terminated with error:" << std::endl;
         std::cout << desc.c_str() << std::endl;
      }
   }

   Falcon::Engine::shutdown();
   return 0;
}
