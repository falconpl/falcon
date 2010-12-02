/**
 *  \file gtk_Main.cpp
 */

#include "gtk_Main.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {


void Main::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Main = mod->addClass( "GtkMain", &Main::init )
        ->addParam( "args" )
        ->addParam( "set_locale" );

    mod->addClassMethod( c_Main, "quit",    &Main::quit );
    mod->addClassMethod( c_Main, "run",     &Main::run ).asSymbol()
       ->addParam( "window" );
}


/*#
    @class GtkMain
    @brief Initialize gtk and prepare the main loop
    @optparam args List of command-line arguments (array of strings)
    @optparam set_locale Set the locale before init (boolean, default true)
    @raise ParamError invalid argument
    @raise GtkError on init failure

    @code
    import from gtk
    m = GtkMain( args, true )
    w = GtkWindow()
    m.run( w )
    @endcode
 */
FALCON_FUNC Main::init( VMARG )
{
    Item* i_args = vm->param( 0 );
    Item* i_setLocale = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( ( i_args && ( i_args->isNil() || !i_args->isArray() ) )
        || ( i_setLocale && ( i_setLocale->isNil() || !i_setLocale->isBoolean() ) ) )
        throw_inv_params( "A,B" );
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
        AutoCString* tmp = NULL;
        char* cstr = NULL;
        const int numargs = getGCharArray( arr, &cstr, &tmp );
        check = gtk_init_check( (int*) &numargs, (char***) &cstr );
        if ( numargs )
        {
            memFree( tmp );
            memFree( cstr );
        }
    }
    else
    {
        int numargs = 0;
        char* cstr[] = { NULL };
        check = gtk_init_check( (int*) &numargs, (char***) &cstr );
    }

    if ( !check )
    {
        throw_gtk_error( e_init_failure, FAL_STR( gtk_e_init_failure_ ) );
    }
}


/*#
    @method quit GtkMain
    @brief Makes the innermost invocation of the main loop return when it regains control.
 */
FALCON_FUNC Main::quit( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    gtk_main_quit();
}


/*#
    @method run GtkMain
    @brief Start the event loop.
    @optparam window A window to be shown
    @raise ParamError Invalid window
 */
FALCON_FUNC Main::run( VMARG )
{
    Item* i_win = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_win && !i_win->isNil()
        && !IS_DERIVED( i_win, GtkWindow ) )
        throw_inv_params( "[GtkWindow]" );
#endif
    if ( i_win )
    {
        GtkWidget* win = (GtkWidget*) COREGOBJECT( i_win )->getObject();
        gtk_widget_show_all( win );
    }

    gtk_main();
}


} // Gtk
} // Falcon
