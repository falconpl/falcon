/**
 *  \file gtk_HButtonBox.cpp
 */

#include "gtk_HButtonBox.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void HButtonBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_HButtonBox = mod->addClass( "GtkHButtonBox", &HButtonBox::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkButtonBox" ) );
    c_HButtonBox->getClassDef()->addInheritance( in );
#if 0
    Gtk::MethodTab methods[] =
    {
    { "get_spacing_default",    &HButtonBox::get_spacing_default },
    { "get_layout_default",     &HButtonBox::get_layout_default },
    { "set_spacing_default",    &HButtonBox::set_spacing_default },
    { "set_layout_default",     &HButtonBox::set_layout_default },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_HButtonBox, meth->name, meth->cb );
#endif
}

/*#
    @class GtkHButtonBox
    @brief Horizontal button box container.

    A button box should be used to provide a consistent layout of buttons throughout
    your application. The layout/spacing can be altered by the programmer,
    or if desired, by the user to alter the 'feel' of a program to a small degree.
 */
FALCON_FUNC HButtonBox::init( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    GtkWidget* wdt = gtk_hbutton_box_new();
    MYSELF;
    Gtk::internal_add_slot( (GObject*) wdt );
    self->setUserData( new GData( (GObject*) wdt ) );
}


//FALCON FUNC get_spacing_default( VMARG );

//FALCON_FUNC get_layout_default( VMARG );

//FALCON_FUNC set_spacing_default( VMARG );

//FALCON_FUNC set_layout_default( VMARG );


} // Gtk
} // Falcon
