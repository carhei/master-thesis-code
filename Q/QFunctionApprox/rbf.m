function phi = rbf(x, xmean, xsigma)

if size(x,2) > 1
    
    xm1 = sum((repmat(x(1),size(xmean,1),1)-xmean(:,1)).^2,2);
    phi1 = exp(-xm1/(2*xsigma(1)^2));
    
    xm2 = sum((repmat(x(2),size(xmean,1),1)-xmean(:,2)).^2,2);
    phi2 = exp(-xm2/(2*xsigma(2)^2));
    
    phi = phi1.*phi2;
    
else
    
    xm = sum((repmat(x,size(xmean,1),1)-xmean).^2,2);
    phi = exp(-xm./(2*xsigma.^2));
end
if sum(phi) > 0
    phi = phi/sum(phi);%*exp(-um/(2*usigma^2));
end
