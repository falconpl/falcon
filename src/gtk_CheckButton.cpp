/**
 *  \file gtk_CheckButton.cpp
 */

#include "gtk_CheckButton.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CheckButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CheckButton = mod->addClass( "GtkCheckButton", &CheckButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkToggleButton" ) );
    c_CheckButton->getClassDef()->addInheritance( in );

    c_CheckButton->setWKS( true );
    c_CheckButton->getClassDef()->factory( &CheckButton::factory );

    mod->addClassMethod( c_CheckButton, "new_with_label",   &CheckButton::new_with_label );
    mod->addClassMethod( c_CheckButton, "new_with_mnemonic",&CheckButton::new_with_mnemonic );
}


CheckButton::CheckButton( const Falcon::CoreClass* gen, const GtkCheckButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* CheckButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new CheckButton( gen, (GtkCheckButton*) btn );
}


/*#
    @class GtkCheckButton
    @brief Create widgets with a discrete toggle button

    A GtkCheckButton places a discrete gtk.ToggleButton next to a widget, (usually
    a GtkLabel). See the section on GtkToggleButton widgets for more information
    about toggle/check buttons.

    The important signal ('toggled') is also inherited from GtkToggleButton.
 */
FALCON_FUNC CheckButton::init( VMARG )
{
    MYSELF;
    if ( self->getGObject() )
        return;
    NO_ARGS
    self->setGObject( (GObject*) gtk_check_button_new() );
}


/*#
    @method new_with_label GtkCheckButton
    @brief Creates a new GtkCheckButton with a GtkLabel to the right of it.
    @param label the text for the check button.
    @return a new GtkCheckButton
 */
FALCON_FUNC CheckButton::new_with_label( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString lbl( i_lbl->asString() );
    GtkWidget* btn = gtk_check_button_new_with_label( lbl.c_str() );
    vm->retval( new Gtk::CheckButton( vm->findWKI( "GtkCheckButton" )->asClass(),
                                      (GtkCheckButton*) btn ) );
}


/*#
    @method new_with_mnemonic GtkCheckButton
    @brief Creates a new GtkCheckButton containing a label.
    @param label The text of the button, with an underscore in front of the mnemonic character
    @return a new GtkCheckButton

    The label will be created using gtk_label_new_with_mnemonic(), so underscores
    in label indicate the mnemonic for the check button.
 */
FALCON_FUNC CheckButton::new_with_mnemonic( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString lbl( i_lbl->asString() );
    GtkWidget* btn = gtk_check_button_new_with_mnemonic( lbl.c_str() );
    vm->retval( new Gtk::CheckButton( vm->findWKI( "GtkCheckButton" )->asClass(),
                                      (GtkCheckButton*) btn ) );
}


} // Gtk
} // Falcon
