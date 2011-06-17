/**
 *  \file gtk_Builder.cpp
 */

#include "gtk_Builder.hpp"
#include "gtk_library.h"

#include <gmodule.h>
#include <ctype.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Builder::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Builder = mod->addClass( "GtkBuilder", &Builder::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_Builder->getClassDef()->addInheritance( in );

    c_Builder->setWKS( true );
    c_Builder->getClassDef()->factory( &Builder::factory );

    Gtk::MethodTab methods[] =
    {
    { "add_from_file",  &Builder::add_from_file },
    { "get_object",     &Builder::get_object },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Builder, meth->name, meth->cb );
}

static GType _falcon_get_type_from_name(GtkBuilder *builder, const gchar *name)
{
    GType type = g_type_from_name( name );

    if( type != G_TYPE_INVALID )
    {
        return type;
    }

    /*
     * Try to map a type name to a _get_type function
     * and call it, eg:
     *
     * GtkWindow -> gtk_window_get_type
     * GtkHBox -> gtk_hbox_get_type
     * GtkUIManager -> gtk_ui_manager_get_type
     *
     */

    // Must be >= 2
    const uint32 reallocSize = 5;
    fassert( reallocSize < 2 );
    const uint32 nameLength = strlen( name );
    /*
      funcNameSize is the length of the underscore-fashioned string plus an extra space.
      realFuncNameSize is funcNameSize plus the length of "_get_type" plus 1 byte for the terminating char.
      Note that the + 1 is omitted because sizeof() counts also the 0-terminating byte at the end.
    */
    uint32 funcNameSize = nameLength + reallocSize;
    uint32 realFuncNameSize = funcNameSize + sizeof( "_get_type" );
    // the output string
    gchar* functionName = (gchar *) memAlloc( sizeof(gchar) * realFuncNameSize );

    // the index for function name
    uint32 pos = 0;
    // capitalize the first letter( dealing with the first character makes the loop simpler )
    if( nameLength )
    {
        functionName[ pos++ ] = tolower( name[0] );
    }

    // the index for name
    uint32 i;
    for( i=1; i < nameLength; i++ )
    {
        gchar ch = name[ i ];

        // If we are on an uppercase character
        if( isupper( ch ) )
        {
            // first check if we have enough space
            if( pos+1 >= funcNameSize )
            {
                funcNameSize += reallocSize;
                realFuncNameSize += reallocSize;
                functionName = (gchar *) memRealloc( functionName, realFuncNameSize );
            }
            // then add an underscore and the character
            functionName[ pos++ ] = '_';
            functionName[ pos++ ] = tolower( ch );

            // if the next character is uppercase as well
            // add & skip it
            if( i+1 < nameLength && isupper( name[ i+1 ] ) )
            {
                functionName[ pos++ ] = tolower( name[ ++i ] );
            }
        }
        // if the character is lowercase just add it
        else
        {
            functionName[ pos++ ] = ch;
        }
    }
    // concatenate with "_get_type"
    strcpy( &functionName[ pos ], "_get_type" );

    GModule* module = g_module_open( GTK2_GTK_LIBRARY, G_MODULE_BIND_LAZY );
    if( module == NULL )
    {
        memFree( functionName );
        return G_TYPE_INVALID;
    }
    
    GType (*funcPtr)();
    
    if( ! g_module_symbol( module, functionName, (gpointer *) &funcPtr ) )
    {
        memFree( functionName );
        g_module_close( module );
        return G_TYPE_INVALID;
    }
    
    g_module_close( module );
    memFree( functionName );
    
    return funcPtr();
}

Builder::Builder( const Falcon::CoreClass *gen, const GtkBuilder *builder ):
    CoreGObject( (CoreClass*) gen, (GObject*) builder )
{}


CoreObject* Builder::factory( const Falcon::CoreClass* gen, void* builder, bool )
{
    return new Builder( gen, (GtkBuilder*) builder );
}


FALCON_FUNC Builder::init( VMARG )
{
    MYSELF;
    if ( self->getObject() )
        return;
    NO_ARGS
    GtkBuilder* builder = gtk_builder_new();
    GtkBuilderClass *c = (GtkBuilderClass*) G_OBJECT_GET_CLASS(builder);
    c->get_type_from_name = _falcon_get_type_from_name;
    self->setObject( (void *) builder );
}

FALCON_FUNC Builder::add_from_file( VMARG )
{
    Item* i_filename = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if(  i_filename == 0
      || ! i_filename->isString() )
    {
        throw_inv_params("S");
    }
#endif
    MYSELF;
    GET_OBJ( self );

    AutoCString filename( i_filename->asString() );
    GError* error = NULL;

    if( ! gtk_builder_add_from_file( GTK_BUILDER(_obj), filename.c_str(), &error ))
    {
        GtkError *err;
        err = new GtkError( ErrorParam( error->code, __LINE__ ).extra( error->message ));
        g_error_free( error );

        throw err;
    }
}


FALCON_FUNC Builder::get_object( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if(  i_name == 0
      || ! i_name->isString() )
    {
        throw_inv_params("S");
    }
#endif

    MYSELF;
    GET_OBJ( self );

    AutoCString name( i_name->asString() );

    GObject* obj = gtk_builder_get_object( GTK_BUILDER(_obj), name.c_str());
    
    if( obj == 0 )
    {
        // TODO: throw an exception
        return;
    }
    
    const char* className = G_OBJECT_TYPE_NAME( obj );

    Item* i_cls = vm->findWKI( className );
    
    if( i_cls == 0 )
    {
        // TODO: throw an exception
        return;
    }

    vm->retval( i_cls->asClass()->createInstance( (void *) obj ) );
}

}
}
