/**
 *  \file gtk_LinkButton.cpp
 */

#include "gtk_LinkButton.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void LinkButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_LinkButton = mod->addClass( "GtkLinkButton", &LinkButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkButton" ) );
    c_LinkButton->getClassDef()->addInheritance( in );

    c_LinkButton->setWKS( true );
    c_LinkButton->getClassDef()->factory( &LinkButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_with_label", &LinkButton::new_with_label },
    { "get_uri",        &LinkButton::get_uri },
    { "set_uri",        &LinkButton::set_uri },
    //{ "set_uri_hook",   &LinkButton::set_uri_hook },
#if GTK_MINOR_VERSION >= 14
    { "get_visited",    &LinkButton::get_visited },
    { "set_visited",    &LinkButton::set_visited },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_LinkButton, meth->name, meth->cb );
}


LinkButton::LinkButton( const Falcon::CoreClass* gen, const GtkLinkButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* LinkButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new LinkButton( gen, (GtkLinkButton*) btn );
}


/*#
    @class GtkLinkButton
    @brief Create buttons bound to a URL
    @optparam uri a valid URI

    A GtkLinkButton is a GtkButton with a hyperlink, similar to the one used by
    web browsers, which triggers an action when clicked. It is useful to show
    quick links to resources.

    A link button is created by calling either gtk_link_button_new() or
    gtk_link_button_new_with_label(). If using the former, the URI you pass to
    the constructor is used as a label for the widget.

    The URI bound to a GtkLinkButton can be set specifically using
    gtk_link_button_set_uri(), and retrieved using gtk_link_button_get_uri().

    GtkLinkButton offers a global hook, which is called when the used clicks on
    it: see gtk_link_button_set_uri_hook().
 */
FALCON_FUNC LinkButton::init( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* uri = args.getCString( 0 );
    MYSELF;
    GtkWidget* wdt = gtk_link_button_new( uri );
    self->setGObject( (GObject*) wdt );
}


/*#
    @method new_with_label
    @brief Creates a new GtkLinkButton containing a label.
    @param uri a valid URI
    @param label the text of the button (or nil).
    @return a new link button widget
 */
FALCON_FUNC LinkButton::new_with_label( VMARG )
{
    Gtk::ArgCheck2 args( vm, "S[,S]" );
    char* uri = args.getCString( 0 );
    char* lbl = args.getCString( 1, false );
    GtkWidget* wdt = gtk_link_button_new_with_label( uri, lbl );
    vm->retval( new Gtk::LinkButton(
            vm->findWKI( "GtkLinkButton" )->asClass(), (GtkLinkButton*) wdt ) );
}


/*#
    @method get_uri
    @brief Retrieves the URI set using gtk_link_button_set_uri().
    @return a valid URI
 */
FALCON_FUNC LinkButton::get_uri( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const char* uri = gtk_link_button_get_uri( (GtkLinkButton*)_obj );
    vm->retval( new String( uri ) );
}


/*#
    @method set_uri
    @brief Sets uri as the URI where the GtkLinkButton points.
    @param uri a valid URI.

    As a side-effect this unsets the 'visited' state of the button.
 */
FALCON_FUNC LinkButton::set_uri( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* uri = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_link_button_set_uri( (GtkLinkButton*)_obj, uri );
}


//FALCON_FUNC LinkButton::set_uri_hook( VMARG );


#if GTK_MINOR_VERSION >= 14
/*#
    @method get_visited
    @brief Retrieves the 'visited' state of the URI where the GtkLinkButton points.
    @return TRUE if the link has been visited, FALSE otherwise

    The button becomes visited when it is clicked. If the URI is changed on the button,
    the 'visited' state is unset again.

    The state may also be changed using gtk_link_button_set_visited().
 */
FALCON_FUNC LinkButton::get_visited( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_link_button_get_visited( (GtkLinkButton*)_obj ) );
}


/*#
    @method set_visited
    @brief Sets the 'visited' state of the URI where the GtkLinkButton points.
    @param visited the new 'visited' state
 */
FALCON_FUNC LinkButton::set_visited( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isInteger() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_link_button_set_visited( (GtkLinkButton*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}
#endif // GTK_MINOR_VERSION >= 14

} // Gtk
} // Falcon
