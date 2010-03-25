/**
 *  \file gtk_Expander.cpp
 */

#include "gtk_Expander.hpp"

#include "gtk_Widget.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Expander::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Expander = mod->addClass( "GtkExpander", &Expander::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_Expander->getClassDef()->addInheritance( in );

    c_Expander->setWKS( true );
    c_Expander->getClassDef()->factory( &Expander::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_with_mnemonic",  &Expander::new_with_mnemonic },
    { "set_expanded",       &Expander::set_expanded },
    { "get_expanded",       &Expander::get_expanded },
    { "set_spacing",        &Expander::set_spacing },
    { "get_spacing",        &Expander::get_spacing },
    { "set_label",          &Expander::set_label },
    { "get_label",          &Expander::get_label },
    { "set_use_underline",  &Expander::set_use_underline },
    { "get_use_underline",  &Expander::get_use_underline },
    { "set_use_markup",     &Expander::set_use_markup },
    { "get_use_markup",     &Expander::get_use_markup },
    { "set_label_widget",   &Expander::set_label_widget },
    { "get_label_widget",   &Expander::get_label_widget },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Expander, meth->name, meth->cb );
}


Expander::Expander( const Falcon::CoreClass* gen, const GtkExpander* exp )
    :
    Gtk::CoreGObject( gen )
{
    if ( exp )
        setUserData( new GData( (GObject*) exp ) );
}


Falcon::CoreObject* Expander::factory( const Falcon::CoreClass* gen, void* exp, bool )
{
    return new Expander( gen, (GtkExpander*) exp );
}


/*#
    @class GtkExpander
    @brief A container which can hide its child
    @optparam label the text of the label

    A GtkExpander allows the user to hide or show its child by clicking on an
    expander triangle similar to the triangles used in a GtkTreeView.

    Normally you use an expander as you would use any other descendant of GtkBin;
    you create the child widget and use gtk_container_add() to add it to the expander.
    When the expander is toggled, it will take care of showing and hiding the
    child automatically.
 */
FALCON_FUNC Expander::init( VMARG )
{
    Gtk::ArgCheck<1> args( vm, "[S]" );

    char* lbl = args.getCString( 0, false );
    if ( !lbl )
        lbl = (char*)"";

    MYSELF;
    GtkWidget* wdt = gtk_expander_new( lbl );
    Gtk::internal_add_slot( (GObject*) wdt );
    self->setUserData( new GData( (GObject*) wdt ) );
}


/*#
    @method new_with_mnemonic GtkExpander
    @brief Creates a new expander using label as the text of the label.
    @param label the text of the label

    If characters in label are preceded by an underscore, they are underlined.
    If you need a literal underscore character in a label, use '__' (two underscores).
    The first underlined character represents a keyboard accelerator called a mnemonic.
    Pressing Alt and that key activates the button.
 */
FALCON_FUNC Expander::new_with_mnemonic( VMARG )
{
    Gtk::ArgCheck<1> args( vm, "S" );

    char* lbl = args.getCString( 0 );

    GtkWidget* wdt = gtk_expander_new_with_mnemonic( lbl );
    vm->retval( new Gtk::Expander(
        vm->findWKI( "GtkExpander" )->asClass(), (GtkExpander*) wdt ) );
}


/*#
    @method set_expanded GtkExpander
    @brief Sets the state of the expander.
    @param expanded whether the child widget is revealed

    Set to true, if you want the child widget to be revealed, and false if you want
    the child widget to be hidden.
 */
FALCON_FUNC Expander::set_expanded( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_expander_set_expanded( (GtkExpander*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_expanded GtkExpander
    @brief Queries a GtkExpander and returns its current state.
    @return the current state of the expander.

    Returns true if the child widget is revealed.
 */
FALCON_FUNC Expander::get_expanded( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_expander_get_expanded( (GtkExpander*)_obj ) );
}


/*#
    @method set_spacing GtkExpander
    @brief Sets the spacing field of expander, which is the number of pixels to place between expander and the child.
    @param distance between the expander and child in pixels.
 */
FALCON_FUNC Expander::set_spacing( VMARG )
{
    Item* i_dist = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_dist || i_dist->isNil() || !i_dist->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_expander_set_spacing( (GtkExpander*)_obj, i_dist->asInteger() );
}


/*#
    @method get_spacing GtkExpander
    @brief Gets the value set by gtk_expander_set_spacing().
    @return spacing between the expander and child.
 */
FALCON_FUNC Expander::get_spacing( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_expander_get_spacing( (GtkExpander*)_obj ) );
}


/*#
    @method set_label GtkExpander
    @brief Sets the text of the label of the expander to label.
    @optparam label a string

    This will also clear any previously set labels.
 */
FALCON_FUNC Expander::set_label( VMARG )
{
    Gtk::ArgCheck<1> args( vm, "S" );

    char* lbl = args.getCString( 0, false );

    MYSELF;
    GET_OBJ( self );
    gtk_expander_set_label( (GtkExpander*)_obj, lbl );
}


/*#
    @method get_label GtkExpander
    @brief Fetches the text from a label widget including any embedded underlines indicating mnemonics and Pango markup, as set by gtk_expander_set_label().
    @return The text of the label widget.

    If the label text has not been set the return value will be NULL. This will
    be the case if you create an empty button with gtk_button_new() to use as a
    container.

    Note that this function behaved differently in versions prior to 2.14 and used
    to return the label text stripped of embedded underlines indicating mnemonics
    and Pango markup. This problem can be avoided by fetching the label text directly
    from the label widget.
 */
FALCON_FUNC Expander::get_label( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const char* lbl = gtk_expander_get_label( (GtkExpander*)_obj );
    if ( lbl )
    {
        String* s = new String( lbl );
        s->bufferize();
        vm->retval( s );
    }
    else
        vm->retnil();
}


/*#
    @method set_use_underline GtkExpander
    @brief If true, an underline in the text of the expander label indicates the next character should be used for the mnemonic accelerator key.
    @param use_underline true if underlines in the text indicate mnemonics
 */
FALCON_FUNC Expander::set_use_underline( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_expander_set_use_underline( (GtkExpander*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_use_underline GtkExpander
    @brief Returns whether an embedded underline in the expander label indicates a mnemonic.
    @return true if an embedded underline in the expander label indicates the mnemonic accelerator keys.
 */
FALCON_FUNC Expander::get_use_underline( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_expander_get_use_underline( (GtkExpander*)_obj ) );
}


/*#
    @method set_use_markup GtkExpander
    @brief Sets whether the text of the label contains markup in Pango's text markup language.
    @param use_markup true if the label's text should be parsed for markup
 */
FALCON_FUNC Expander::set_use_markup( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_expander_set_use_markup( (GtkExpander*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_use_markup GtkExpander
    @brief Returns whether the label's text is interpreted as marked up with the Pango text markup language.
    @return true if the label's text will be parsed for markup
 */
FALCON_FUNC Expander::get_use_markup( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_expander_get_use_markup( (GtkExpander*)_obj ) );
}


/*#
    @method set_label_widget GtkExpander
    @brief Set the label widget for the expander. This is the widget that will appear embedded alongside the expander arrow.
    @param label_widget the new label widget (or nil)
 */
FALCON_FUNC Expander::set_label_widget( VMARG )
{
    Gtk::ArgCheck<0> args( vm, "[GtkWidget]" );
    // this method accepts nil
    GtkWidget* wdt = NULL;
    CoreObject* o_wdt = args.getObject( 0, false );
    if ( o_wdt && !CoreObject_IS_DERIVED( o_wdt, GtkWidget ) )
        throw_inv_params( "[GtkWidget]" );
    else
    if ( o_wdt )
        wdt = (GtkWidget*)((GData*)o_wdt->getUserData())->obj();

    MYSELF;
    GET_OBJ( self );
    gtk_expander_set_label_widget( (GtkExpander*)_obj, wdt );
}


/*#
    @method get_label_widget GtkExpander
    @brief Retrieves the label widget for the frame.
    @return the label widget, or nil if there is none
 */
FALCON_FUNC Expander::get_label_widget( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_expander_get_label_widget( (GtkExpander*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


} // Gtk
} // Falcon
