/**
 *  \file gtk_Main.cpp
 */

#include "gtk_Main.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {


void Main::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Main = mod->addClass( "Main", &Main::init )
        ->addParam( "args" )
        ->addParam( "set_locale" );

    mod->addClassMethod( c_Main, "quit",    &Main::quit );
    mod->addClassMethod( c_Main, "run",     &Main::run ).asSymbol()
       ->addParam( "window" );
}


/*#
    @class gtk.Main
    @brief Initialize gtk and prepare the main loop
    @optparam args List of command-line arguments (array of strings)
    @optparam set_locale Set the locale before init (boolean, default true)
    @raise ParamError invalid argument
    @raise GtkError on init failure

    @code
    import from gtk
    m = gtk.Main( args, true )
    w = gtk.Window()
    m.run( w )
    @endcode
 */

/*#
    @init gtk.Main
 */
FALCON_FUNC Main::init( VMARG )
{
    Item* i_args = vm->param( 0 );
    Item* i_setLocale = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( ( i_args && ( i_args->isNil() || !i_args->isArray() ) )
        || ( i_setLocale && ( i_setLocale->isNil() || !i_setLocale->isBoolean() ) ) )
    {
        throw_inv_params( "A,B" );
    }
#endif
    bool setLocale = true;

    if ( i_setLocale )
        setLocale = i_setLocale->asBoolean();

    /*
     *  disable locale if requested, before calling init()
     */
    if ( !setLocale )
        gtk_disable_setlocale();

    /*
     *  init gtk
     */

    gboolean check = false;

    if ( i_args )
    {
        CoreArray* arr = i_args->asArray();
        const int numargs = arr->length();
        AutoCString* strings = (AutoCString*) memAlloc(
            sizeof( AutoCString ) * numargs );
        char** cstrings = (char**) memAlloc(
            sizeof( char** ) * ( numargs + 1 ) );
        cstrings[ numargs ] = NULL; // just in case
        Item* it;

        for ( int i=0; i < numargs; ++i )
        {
            it = &arr->at( i );
#ifndef NO_PARAMETER_CHECK
            if ( !it->isString() )
            {
                throw_inv_params( "S" );
            }
#endif
            strings[i] = AutoCString( it->asString() );
            cstrings[i] = (char*) strings[i].c_str();
        }

        check = gtk_init_check( (int*) &numargs, (char***) &cstrings );

        memFree( strings );
        memFree( cstrings );
    }
    else
    {
        int numargs = 0;
        char* cstrings[] = { NULL };
        check = gtk_init_check( (int*) &numargs, (char***) &cstrings );
    }

    if ( !check )
    {
        throw_gtk_error( e_init_failure, FAL_STR( gtk_e_init_failure_ ) );
    }
}

/*#
    @method quit gtk.Main
    @brief Makes the innermost invocation of the main loop return when it regains control.
 */
FALCON_FUNC Main::quit( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    gtk_main_quit();
}

/*#
    @method run gtk.Main
    @brief Start the event loop.
    @optparam window A window to be shown
    @raise ParamError Invalid window
 */
FALCON_FUNC Main::run( VMARG )
{
    Item* i_win = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_win && !i_win->isNil()
        && !i_win->isOfClass( "gtk.Window" ) && !i_win->isOfClass( "Window" ) )
    {
        throw_inv_params( "[Window]" );
    }
#endif
    if ( i_win )
    {
        CoreObject* o_win = i_win->asObject();
        GtkWidget* win = (GtkWidget*)((GData*)o_win->getUserData())->obj();

        gtk_widget_show_all( win );
    }

    gtk_main();
}

} // Gtk
} // Falcon
