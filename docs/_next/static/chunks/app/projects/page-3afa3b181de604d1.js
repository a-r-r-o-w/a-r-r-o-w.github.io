(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[895],{473:function(e,s,n){Promise.resolve().then(n.bind(n,5314))},5314:function(e,s,n){"use strict";n.r(s),n.d(s,{default:function(){return i}});var r=n(7437),a=n(2265),t=n(6691),l=n.n(t),c=function(){arguments.length>0&&void 0!==arguments[0]&&arguments[0];let e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:-1;return(0,r.jsx)("img",{className:"missing-image-gradient cards__img",style:{"--gradient-angle":e%2==0?"45deg":"135deg"}})};function i(e){let{url:s,name:n,description:t,languages:i,image:d,video:u,index:o}=e,[m,_]=(0,a.useState)(!1),h=(0,a.useRef)(null);(0,a.useEffect)(()=>{_(""!==document.createElement("video").canPlayType("video/mp4"))},[]);let g=c(n,o);return null!==d&&(g=(0,r.jsx)(l(),{className:"cards__img",src:d,alt:n,width:300,height:300})),null!==u&&m&&(g=(0,r.jsx)("video",{className:"cards__vid",ref:h,loop:!0,muted:!0,playsInline:!0,alt:n,src:u,children:"Your browser does not support the video tag."})),(0,r.jsx)("div",{className:"cards__project",onMouseEnter:()=>{null!==h.current&&h.current.play()},onMouseLeave:()=>{null!=h.current&&(h.current.pause(),h.current.currentTime=0)},children:(0,r.jsxs)("a",{href:s,children:[(0,r.jsx)(a.Suspense,{fallback:c(o),children:g}),(0,r.jsxs)("div",{className:"cards__text",children:[(0,r.jsx)("div",{className:"font-bold text-xl;",children:n}),(0,r.jsx)("div",{className:"text-base",id:"description",children:t}),(0,r.jsx)("div",{className:"cards__languages",children:i.map((e,s)=>(0,r.jsx)("span",{className:"cards__language_container",children:e},s))})]})]})})}}},function(e){e.O(0,[986,971,938,744],function(){return e(e.s=473)}),_N_E=e.O()}]);