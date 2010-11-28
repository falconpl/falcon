/**
 *  \file gtk_Invisible.cpp
 */

#include "gtk_Invisible.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Invisible::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Invisible = mod->addClass( "GtkInvisible", &Invisible::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkWidget" ) );
    c_Invisible->getClassDef()->addInheritance( in );

    c_Invisible->getClassDef()->factory( &Invisible::factory );

    Gtk::MethodTab methods[] =
    {
#if 0
    { "new_for_screen",     &Invisible::new_for_screen },
    { "set_screen",         &Invisible::set_screen },
    { "get_screen",         &Invisible::get_screen },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Invisible, meth->name, meth->cb );
}


Invisible::Invisible( const Falcon::CoreClass* gen, const GtkInvisible* arrow )
    :
    Gtk::CoreGObject( gen, (GObject*) arrow )
{}


Falcon::CoreObject* Invisible::factory( const Falcon::CoreClass* gen, void* arrow, bool )
{
    return new Invisible( gen, (GtkInvisible*) arrow );
}


/*#
    @class GtkInvisible
    @brief A widget which is not displayed

    The GtkInvisible widget is used internally in GTK+, and is probably not very useful
    for application developers.

    It is used for reliable pointer grabs and selection handling in the code for
    drag-and-drop.
 */
FALCON_FUNC Invisible::init( VMARG )
{
    NO_ARGS
    MYSELF;
    GtkWidget* wdt = gtk_invisible_new();
    self->setObject( (GObject*) wdt );
}

#if 0
FALCON_FUNC Invisible::new_for_screen( VMARG );

FALCON_FUNC Invisible::set_screen( VMARG );

FALCON_FUNC Invisible::get_screen( VMARG );
#endif

} // Gtk
} // Falcon
