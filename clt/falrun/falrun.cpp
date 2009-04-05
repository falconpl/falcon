/*
   FALCON - The Falcon Programming Language.
   FILE: falrun.cpp

   A simple program that uses Falcon VM to execute falcon compiled codes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago  8 20:30:13 CEST 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vm.h>
#include <falcon/sys.h>
#include <falcon/modloader.h>
#include <falcon/runtime.h>
#include <falcon/core_ext.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/string.h>
#include <falcon/stdstreams.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/fassert.h>
#include <falcon/genericlist.h>


using namespace Falcon;

List preloaded;

String load_path;
String module_language;
Stream *stdIn;
Stream *stdOut;
Stream *stdErr;
String io_encoding;
bool ignore_defpath = false;


static void version()
{
   stdOut->writeString( "FALCON runtime virtual machine.\n" );
   stdOut->writeString( "Version " );
   stdOut->writeString( FALCON_VERSION " (" FALCON_VERSION_NAME ")" );
   stdOut->writeString( "\n" );
}

static void usage()
{
   stdOut->writeString( "Usage: falrun [options] file.hsc [script options]\n" );
   stdOut->writeString( "\n" );
   stdOut->writeString( "Options:\n" );
   stdOut->writeString( "   -e <enc>    select default encoding for VM streams\n" );
   stdOut->writeString( "   -h          this help\n" );
   stdOut->writeString( "   -p mod      pump one or more module in the virtual machine\n" );
   stdOut->writeString( "   -l <lang>   Set preferential language of loaded modules\n" );
   stdOut->writeString( "   -L<path>    set path for 'load' directive\n" );
   stdOut->writeString( "   -P          Ignore default system paths\n" );
   stdOut->writeString( "   -v          print copyright notice and version and exit\n" );
   stdOut->writeString( "\n" );
   stdOut->writeString( "Paths must be in falcon file name format: directory separatros must be slashes and\n" );
   stdOut->writeString( "multiple entries must be entered separed by a semicomma (';')\n" );
   stdOut->writeString( "\n" );
}

void findModuleName( const String &filename, String &name )
{
   uint32 pos = filename.rfind( "/" );
   if ( pos == csh::npos ) {
      name = filename;
   }
   else {
      name = filename.subString( pos + 1 );
   }

   // find the dot
   pos = name.rfind( "." );
   if ( pos != csh::npos ) {
      name = name.subString( 0, pos );
   }
}

void findModulepath( const String &filename, String &path )
{
   uint32 pos = filename.rfind( "/" );
   if ( pos != csh::npos ) {
      path = filename.subString( 0, pos );
   }
}


String get_load_path()
{
   String envpath;
   bool hasEnvPath = Sys::_getEnv ( "FALCON_LOAD_PATH", envpath );

   String lp;

   if ( load_path.size() > 0 )
   {
      if ( ignore_defpath )
         lp = load_path;
      else
         lp = FALCON_DEFAULT_LOAD_PATH ";" + load_path;
   }

   if ( ! hasEnvPath || ignore_defpath )
   {
      return lp;
   }
   else
   {
      if ( lp == "" )
         return envpath;
      else
         return lp + ";" + envpath;
   }
}


String get_io_encoding()
{
   if ( io_encoding != "" )
      return io_encoding;

   String ret;
   if ( Sys::_getEnv("FALCON_VM_ENCODING", ret) )
      return ret;

   return "";
}


int main( int argc, char *argv[] )
{
   // Install a void ctrl-c handler (let ctrl-c to kill this app)
   Sys::_dummy_ctrl_c_handler();

   Falcon::Engine::AutoInit autoInit;


   int script_pos = argc;
   char *input_file = 0;
   FileStream *bincode_stream;

   stdOut = stdOutputStream();
   stdErr = stdErrorStream();
   stdIn = stdInputStream();

   // option decoding
   for ( int i = 1; i < argc; i++ )
   {
      char *op = argv[i];

      if (op[0] == '-' )
      {
         switch ( op[1] )
         {
            case 'e':
               if ( op[2] == 0 && i < argc + 1) {
                  io_encoding = argv[++i];
               }
               else {
                  io_encoding = op + 2;
               }
            break;

            case 'h': usage(); return 0;

            case 'L':
               if ( op[2] == 0 && i < argc + 1)
                  load_path = argv[++i];
               else
                  load_path = op + 2; break;
            break;

            case 'P':
               ignore_defpath = true;
               break;

            case 'l':
               if ( op[2] == 0 && i + 1 < argc )
                  module_language = argv[++i];
               else
                  module_language = op + 2;
            break;

            case 'p':
               if ( op[2] == 0 && i < argc + 1)
                  preloaded.pushBack( argv[++i] );
               else
                  preloaded.pushBack( op + 2 );
            break;


            case 'v': version(); return 0;

            default:
               stdOut->writeString( "falrun: unrecognized option '" );
               stdOut->writeString( op );
               stdOut->writeString( "'.\n\n" );
               usage();
               return 1;
         }
      }
      else {
         input_file = op;
         script_pos = i+1;
         break;
      }
   }

   // eventually change the encodings.
   io_encoding = get_io_encoding();

   if ( io_encoding != "" )
   {
      Transcoder *trans = TranscoderFactory( io_encoding, 0, true );
      if ( trans == 0 )
      {
         stdOut->writeString( "Fatal: unrecognized encoding '" + io_encoding + "'.\n\n" );
         return 1;
      }
      delete stdIn ;
      delete stdOut;
      delete stdErr;

      trans->setUnderlying( new StdInStream );

      stdIn = AddSystemEOL( trans, true );
      stdOut = AddSystemEOL( TranscoderFactory( io_encoding, new StdOutStream, true ), true );
      stdErr = AddSystemEOL( TranscoderFactory( io_encoding, new StdErrStream, true ), true );
   }

   if ( input_file == 0 ) {
      stdOut->writeString( "falrun: missing script name.\n" );
      usage();
      return 1;
   }

   bincode_stream = new FileStream;
   bincode_stream->open( input_file );

   if ( ! bincode_stream->good() )
   {
      stdOut->writeString( "falrun: Can't open file " );
      stdOut->writeString( input_file );
      stdOut->writeString( "\n" );
      return 1;
   }


   String module_name;
   String source_path;
   findModuleName( input_file, module_name );
   findModulepath( input_file, source_path );

   //-----------------------------------------
   // execute the script.
   //

   if ( source_path != "" )
      source_path += ";";

   try
   {
      ModuleLoader *modloader = new ModuleLoader( source_path + get_load_path() );

      // set the module preferred language; ok also if default ("") is used
      modloader->setLanguage( module_language );

      Module *core = core_module_init();

      Module *main_mod = modloader->loadModule( bincode_stream );

      VMachine *vmachine = new VMachine(false);
      // change default machine streams.
      vmachine->stdIn( stdIn );
      vmachine->stdOut( stdOut );
      vmachine->stdErr( stdErr );
      vmachine->init();

      vmachine->link( core );
      core->decref();
      Runtime *runtime = new Runtime( modloader );

      // preload required modules

      ListElement *pliter = preloaded.begin();
      while( pliter != 0 )
      {
         Module *module = modloader->loadName( * ((String *) pliter->data()) );
         runtime->addModule( module );
         pliter = pliter->next();
      }

      Item *item_args = vmachine->findGlobalItem( "args" );
      fassert( item_args != 0 );
      CoreArray *args = new CoreArray( argc - script_pos );
      for ( int ap = script_pos; ap < argc; ap ++ ) {
         args->append( new CoreString( argv[ap] ) );
      }
      item_args->setArray( args );

      Item *script_name = vmachine->findGlobalItem( "scriptName" );
      fassert( script_name != 0 );
      script_name->setString( new CoreString( module_name ) );

      // the runtime will try to load the references.
      runtime->addModule( main_mod );

      if( vmachine->link( runtime ) && vmachine->launch() )
      {
         if ( vmachine->regA().type() == FLC_ITEM_INT )
            return (int32) vmachine->regA().asInteger();
         else
            return 0;
      }

      vmachine->finalize();
   }
   catch ( Error *err )
   {
      String temp;
      err->toString( temp );
      stdErr->writeString( "falcon: FATAL - Program terminated with error.\n" );
      stdErr->writeString( temp + "\n" );
      err->decref();
      return 1;
   }


   return 255;
}


/* end of falrun.cpp */
