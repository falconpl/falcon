/**
 *  \file gtk_VButtonBox.cpp
 */

#include "gtk_VButtonBox.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void VButtonBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_VButtonBox = mod->addClass( "VButtonBox", &VButtonBox::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "ButtonBox" ) );
    c_VButtonBox->getClassDef()->addInheritance( in );
#if 0
    Gtk::MethodTab methods[] =
    {
    { "get_spacing_default",    &VButtonBox::get_spacing_default },
    { "get_layout_default",     &VButtonBox::get_layout_default },
    { "set_spacing_default",    &VButtonBox::set_spacing_default },
    { "set_layout_default",     &VButtonBox::set_layout_default },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_VButtonBox, meth->name, meth->cb );
#endif
}

/*#
    @class gtk.VButtonBox
    @brief Vertical button box container.

    A button box should be used to provide a consistent layout of buttons throughout
    your application. The layout/spacing can be altered by the programmer,
    or if desired, by the user to alter the 'feel' of a program to a small degree.
 */

/*#
    @init gtk.VButtonBox
    @brief Creates a new horizontal button box.
 */
FALCON_FUNC VButtonBox::init( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    GtkWidget* wdt = gtk_vbutton_box_new();
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
